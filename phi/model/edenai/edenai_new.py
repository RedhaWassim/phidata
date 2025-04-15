from time import time
import requests
import json
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Dict, Any, AsyncIterator, Union, Tuple

from phi.model.base import Model
from phi.model.message import Message
from phi.model.response import ModelResponse
from phi.tools.function import FunctionCall
from phi.utils.log import logger
from phi.utils.timer import Timer
from os import getenv
from pydantic import BaseModel
from phi.utils.tools import get_function_call_for_tool_call
import aiohttp
import httpx
def parse_edenai_stream_chunk(chunk_str: str) -> Tuple[Optional[str], Optional[Dict]]:
    """Parse EdenAI's streaming response chunks into (delta, raw_chunk) tuples"""
    if not chunk_str.strip():
        return None, None

    try:
        chunk = json.loads(chunk_str)
        if "choices" in chunk and isinstance(chunk["choices"], list):
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            return content, chunk
        return None, chunk
    except json.JSONDecodeError:
        return None, {"raw_chunk": chunk_str}
    except Exception as e:
        return None, {"error": str(e)}
@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def log(self):
        logger.debug("**************** METRICS START ****************")
        if self.time_to_first_token is not None:
            logger.debug(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        logger.debug(f"* Time to generate response:   {self.response_timer.elapsed:.4f}s")
        logger.debug(f"* Input tokens:                {self.input_tokens}")
        logger.debug(f"* Output tokens:               {self.output_tokens}")
        logger.debug(f"* Total tokens:                {self.total_tokens}")
        logger.debug("**************** METRICS END ******************")

class EdenAIChat(Model):
    id: str = "edenai-chat"
    name: str = "openai/gpt-4o"
    api_key: Optional[str] = getenv("EDENAI_API_KEY")
    base_url: str = "https://api.edenai.run/v2/llm/chat/completions"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    tools: Optional[List[Dict]] = None

    def _get_request_params(self) -> Dict:
        params = {
            "model": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.stream
        }
        if self.tools:
            params["tools"] = self.tools
        return {k: v for k, v in params.items() if v is not None}

    def _format_messages_openai(self, messages: List[Message]) -> List[Dict]:
        formatted_messages = []
        for msg in messages:
            message = {"role": msg.role, "content": msg.content}
            if self.tools:
                message["tool_calls"] = self.tools
            formatted_messages.append(message)


        print("**********")
        print(formatted_messages)
        print("************")
        return formatted_messages

    def invoke(self, messages: List[Message]) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            **self._get_request_params(),
            "messages": self._format_messages_openai(messages)
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"EdenAI API Error: {str(e)}")
            raise

    async def ainvoke(self, messages: List[Message]) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            **self._get_request_params(),
            "messages": self._format_messages_openai(messages)
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.error(f"EdenAI Async API Error: {str(e)}")
            raise

    # Streaming implementation
    def invoke_stream(self, messages: List[Message]) -> Iterator[Dict]:
        """Sync streaming implementation"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.name,
            "messages": self._format_messages_openai(messages),
            "stream": True,
        }

        with httpx.Client() as client:
            with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                for line in response.iter_lines():
                    delta, raw_chunk = parse_edenai_stream_chunk(line)
                    if delta is None:
                        continue
                    if raw_chunk is not None:
                        yield {"delta": delta, "raw": raw_chunk}

    async def ainvoke_stream(self, messages: List[Message]) -> AsyncIterator[Dict]:
        """Async streaming implementation"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.name,
            "messages": self._format_messages_openai(messages),
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.base_url, headers=headers, json=payload) as response:
                async for line in response.aiter_lines():
                    delta, raw_chunk = parse_edenai_stream_chunk(line)
                    if raw_chunk is not None:
                        yield {"delta": delta, "raw": raw_chunk}

    def _parse_response(self, raw_response: Dict, stream : bool = False) -> ModelResponse:
        model_response = ModelResponse()
        
        if "choices" in raw_response and len(raw_response["choices"]) > 0:
            choice = raw_response["choices"][0]
            message = choice.get("message", {})
            
            model_response.content = message.get("content")
            if self.tools != [] : 
                model_response.tool_call = self._parse_tool_calls(raw_response, stream)
            print("////////////")
            print(model_response.tool_call)
            print("////////////")
            if "usage" in raw_response:
                usage = raw_response["usage"]
                model_response.metrics = {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens")
                }
        
        return model_response

    def _parse_stream_chunk(self, chunk: Dict) -> ModelResponse:
        model_response = ModelResponse()
        raw_chunk = chunk.get("raw", {})

        model_response.delta = chunk.get("delta", "")
        model_response.content = model_response.delta
        model_response.tool_call = self._parse_tool_calls(raw_chunk, stream = True)

        return model_response

    def _parse_tool_calls(self, response_data: Dict, stream: bool = False) -> List[Dict]:
        """Parse tool calls from EdenAI response structure"""
        tool_calls = []
        try : 
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]

                if stream:
                    delta = choice.get("delta", {})
                    if "tool_calls" in delta:
                        for tool_call in delta["tool_calls"]:
                            parsed_call = {
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get("arguments")
                                }
                            }
                            tool_calls.append(parsed_call)
                else:
                    message = choice.get("message", {})
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            parsed_call = {
                                "id": tool_call.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get("arguments")
                                }
                            }
                            tool_calls.append(parsed_call)

            return tool_calls
        except :
            return None
    def handle_tool_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
        tool_role: str = "tool",
    ) -> Optional[ModelResponse]:
        """Process tool calls in assistant message"""
        if not assistant_message.tool_calls:
            return None

        if model_response.content is None:
            model_response.content = ""

        function_call_results = []
        function_calls_to_run = []

        print("----------")
        print(assistant_message.tool_calls)
        print("-----------")
        
        for tool_call in assistant_message.tool_calls:
            function_call = get_function_call_for_tool_call(tool_call, self.tools)
            
            if not function_call:
                messages.append(Message(
                    role=tool_role,
                    tool_call_id=tool_call.get("id"),
                    content="Could not find function to call."
                ))
                continue

            if function_call.error:
                messages.append(Message(
                    role=tool_role,
                    tool_call_id=tool_call.get("id"),
                    content=function_call.error
                ))
                continue

            function_calls_to_run.append(function_call)

        if self.show_tool_calls:
            model_response.content += "\nTool calls:"
            for func in function_calls_to_run:
                model_response.content += f"\n- {func.get_call_str()}"
            model_response.content += "\n"

        # Execute tool calls
        for _ in self.run_function_calls(function_calls_to_run, function_call_results):
            pass

        if function_call_results:
            messages.extend(function_call_results)

        return model_response

    def response(self, messages: List[Message]) -> ModelResponse:
        try:
            raw_response = self.invoke(messages)
            model_response = self._parse_response(raw_response)
            
            # Create assistant message with tool calls
            assistant_message = Message(
                role="assistant",
                content=model_response.content,
                tool_calls=model_response.tool_call
            )
            messages.append(assistant_message)

            # Handle tool calls if any
            if model_response.tool_call:
                return self.handle_tool_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=model_response
                )
            
            return model_response
            
        except Exception as e:
            return ModelResponse(error=str(e))
    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        try:
            # Get raw response from async invocation
            raw_response = await self.ainvoke(messages)
            print(raw_response)
            # Parse using unified response parser
            model_response = self._parse_response(raw_response, stream=False)
            print("***************")
            
            print(model_response)
            print("*************")
            # Create assistant message with parsed content and tool calls
            assistant_message = Message(
                role="assistant",
                content=model_response.content,
                tool_calls=model_response.tool_call
            )
            messages.append(assistant_message)

            print("-----------------------")
            print(assistant_message)
            print("----------------")
            # Handle tool calls if present
            if model_response.tool_call:
                return self.handle_tool_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=model_response
                )
            
            return model_response
            
        except Exception as e:
            return ModelResponse(content=f"Error: {str(e)}")
    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        full_response = ModelResponse()
        print("HERRREEE")
        try:
            for chunk in self.invoke_stream(messages):
                print(chunk)
                chunk_response = self._parse_stream_chunk(chunk)

                # Accumulate content
                if chunk_response.delta:
                    full_response.content = (full_response.content or "") + chunk_response.delta

                # Handle tool calls if present in chunk
                if hasattr(chunk_response, 'tool_calls') and chunk_response.tool_call:
                    if not hasattr(full_response, 'tool_calls'):
                        full_response.tool_call = []
                    full_response.tool_call.extend(chunk_response.tool_call)

                yield chunk_response

            # After stream completes, handle tool calls if any
            if hasattr(full_response, 'tool_calls') and full_response.tool_call:
                assistant_message = Message(
                    role="assistant",
                    content=full_response.content,
                    tool_calls=full_response.tool_call
                )
                messages.append(assistant_message)

                self.handle_tool_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=full_response
                )

        except Exception as e:
            # Use proper ModelResponse initialization
            yield ModelResponse(content=f"Error: {str(e)}")
    async def aresponse_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        full_response = ModelResponse()
        
        try:
            async for chunk in self.ainvoke_stream(messages):
                chunk_response = self._parse_stream_chunk(chunk)
                
                if chunk_response.delta:
                    full_response.content = (full_response.content or "") + chunk_response.delta
                
                if chunk_response.tool_call:
                    full_response.tool_call = chunk_response.tool_call
                
                yield chunk_response

            if full_response.tool_call:
                assistant_message = Message(
                    role="assistant",
                    content=full_response.content,
                    tool_calls=full_response.tool_call
                )
                messages.append(assistant_message)
                
                self.handle_tool_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    model_response=full_response
                )
                
        except Exception as e:
            yield ModelResponse(error=str(e))