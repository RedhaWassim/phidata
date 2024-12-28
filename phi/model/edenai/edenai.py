from time import time
import requests
import json
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Dict, Any, AsyncIterator

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

class EdenaiChatMessage:
    generated_text: str
    tools : Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def __init__(self, generated_text: str, tools: Optional[str], tool_calls: Optional[List[Dict[str, Any]]] = None):
        self.generated_text = generated_text
        self.tools = tools
        self.tool_calls = tool_calls
        if self.tool_calls is not None:
            for tool in self.tool_calls:
                tool["type"] = "function"
                tool["function"] = {"name": tool["name"], "arguments": tool["arguments"]}

class EdenAIChat(Model):
    """
    Class for interacting with the EdenAI API.

    Attributes :
    - id: str: The unique identifier of the model. Default: "deepseek-chat".
    - name: str: The name of the model you want to use, for the list of available models visit : https://www.edenai.co
    - api_key: Optional[str]:The API key for the model
    - base_url: str: The base URL for the model. Default: "https://api.edenai.run/v2/text/chat".

    """

    id: str = "edenai-chat" 
    name: str = "openai/gpt-4o"

    api_key: Optional[str] = getenv("EDENAI_API_KEY", None) 
    base_url: str = "https://api.edenai.run/v2/text/chat"

    def convert_tools_to_api_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert the tools to the format expected by the EdenAI API.

        Args:
            tools (List[Dict[str, Any]]): The list of tools.

        Returns:
            List[Dict[str, Any]]: The tools in the format expected by the EdenAI API.
        """
        api_tools = []
        for tool in tools:
            api_tool = tool["function"]
            api_tools.append(api_tool)
        return api_tools

    def handle_tool_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
        tool_role: str = "tool",
    ) -> Optional[ModelResponse]:
        """
        Handle tool calls in the assistant message.

        Args:
            assistant_message (Message): The assistant message.
            messages (List[Message]): The list of messages.
            model_response (ModelResponse): The model response.
            tool_role (str): The role of the tool call. Defaults to "tool".

        Returns:
            Optional[ModelResponse]: The model response after handling tool calls.
        """
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0:
            if model_response.content is None:
                model_response.content = ""
            function_call_results: List[Message] = []
            function_calls_to_run: List[FunctionCall] = []

            for tool_call in assistant_message.tool_calls:
                tool_call_id = tool_call.get("id")
                function_call = get_function_call_for_tool_call(tool_call,self.functions)
                if function_call is None:
                    messages.append(
                        Message(
                            role=tool_role,
                            tool_call_id=tool_call_id,
                            content="Could not find function to call.",
                        )
                    )
                    continue

                if function_call.error is not None:
                    messages.append(
                        Message(
                            role=tool_role,
                            tool_call_id=tool_call_id,
                            content=function_call.error,
                        )
                    )
                    continue

                function_calls_to_run.append(function_call)

            if self.show_tool_calls:
                model_response.content += "\nRunning tool calls:"
                for func in function_calls_to_run:
                    model_response.content += f"\n - {func.get_call_str()}"
                model_response.content += "\n\n"

            for _ in self.run_function_calls(
                function_calls=function_calls_to_run, function_call_results=function_call_results
            ):
                pass

            if len(function_call_results) > 0:
                messages.extend(function_call_results)

            return model_response
        return None

    def format_messages(self, messages: List[Message]) -> str:
        """
        Format messages to match the EdenAI API structure for 'text' input.

        Args:
            messages: List of Message objects.

        Returns:
            str: The formatted text to be sent to EdenAI API.
        """
        formatted_text = ""

        for msg in messages:
            formatted_text += f"{msg.content}\n"

        return formatted_text.strip() 
    
    def format_available_tools(self, tools):
        formatted_tools = []

        for tool in tools:
            if "function" in tool:
                function = tool["function"]

                formatted_tool = {
                    "type": "object",
                    "properties": {}
                }

                for param_name, param_details in function.get("parameters", {}).get("properties", {}).items():
                    formatted_tool["properties"][param_name] = {
                        "type": param_details.get("type", ""),
                        "description": param_details.get("description", ""),
                        "enum": param_details.get("enum", [])
                    }

                formatted_tool["required"] = function.get("parameters", {}).get("required", [])
                formatted_tools.append(formatted_tool)

        return formatted_tools


    def invoke(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Invoke the EdenAI API with the given messages and tools.

        Args:
            messages: List of Message objects.

        Returns:
            Dict[str, Any]: The raw response from the EdenAI API.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "providers": [self.name],
            "text": self.format_messages(messages),
        }
        if self.tools:
            function_tools = self.convert_tools_to_api_format(self.tools)
            payload["available_tools"] = function_tools
        metrics = Metrics()
        metrics.response_timer.start()

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            metrics.response_timer.stop()
            if response.status_code != 200:
                logger.error(f"EdenAI Error: {response.text}")
                raise ValueError(f"EdenAI API Error: {response.text}")
            return response.json()

        except Exception as e:
            logger.error(f"Error invoking EdenAI API: {e}")
            metrics.response_timer.stop()
            raise

    def create_assistant_message(
        self,
        response_message: EdenaiChatMessage,
        metrics: Metrics,
        response_usage: Optional[float],
    ) -> Message:
        """
        Create an assistant message from the response.

        Args:
            response_message (ChatCompletionMessage): The response message.
            metrics (Metrics): The metrics.
            response_usage (Optional[CompletionUsage]): The response usage.

        Returns:
            Message: The assistant message.
        """


        assistant_message = Message(
            role="assistant",
            content=response_message.generated_text,
        )
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            try:
                assistant_message.tool_calls = response_message.tool_calls
            except Exception as e:
                logger.warning(f"Error processing tool calls: {e}")
        self.update_usage_metrics(assistant_message, metrics, response_usage)
        return assistant_message


    def response(self, messages: List[Message]) -> ModelResponse:
        """
        Send a generate content request to EdenAI and return the response.

        Args:
            messages (List[Message]): The list of messages to send to the model.

        Returns:
            ModelResponse: The model response.
        """
        logger.debug("---------- EdenAI Response Start ----------")
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()
        metrics.response_timer.start()
        try:
            raw_response = self.invoke(messages=messages)[self.name]  
            metrics.response_timer.stop()
        except Exception as e:
            logger.error(f"Error in EdenAI response: {e}")
            metrics.response_timer.stop()
            raise

        for msg in raw_response["message"]:
            if msg["role"] == "assistant":
                response_message = EdenaiChatMessage(
                generated_text=msg.get("message", ""),
                tools=msg.get("tools", ""),
                tool_calls=msg.get("tool_calls", []),
                )
        response_usage = {
            "input_token": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

        try:
            if (
                self.response_format is not None
                and self.structured_outputs
                and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.parsed
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            logger.warning(f"Error retrieving structured outputs: {e}")

        assistant_message = self.create_assistant_message(
            response_message=response_message, metrics=metrics, response_usage=response_usage
        )

        messages.append(assistant_message)

        assistant_message.log()
        metrics.log()

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.audio is not None:
            model_response.audio = assistant_message.audio

        tool_role = "tool"
        if (
            self.handle_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                model_response=model_response,
                tool_role=tool_role,
            )
            is not None
        ):
            return self.handle_post_tool_call_messages(messages=messages, model_response=model_response)
        logger.debug("---------- OpenAI Response End ----------")
        return model_response
    

    def update_usage_metrics(
        self, assistant_message: Message, metrics: Metrics, usage: Optional[Dict[str, Any]] = None
    ) -> None:
        try : 
            """
            Update usage metrics for the assistant message.

            Args:
                assistant_message: Message object containing the response content.
                metrics: Metrics object containing the usage metrics.
                usage: Dict containing usage data.
            """
            assistant_message.metrics["time"] = metrics.response_timer.elapsed
            self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)

            if usage:
            
                metrics.input_tokens = usage.get("input_tokens", 0)
                metrics.output_tokens = usage.get("output_tokens", 0)
                metrics.total_tokens = usage.get("total_tokens", 0)

                assistant_message.metrics["input_tokens"] = metrics.input_tokens
                assistant_message.metrics["output_tokens"] = metrics.output_tokens
                assistant_message.metrics["total_tokens"] = metrics.total_tokens

                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + metrics.input_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + metrics.output_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens
        except Exception as e:
            logger.error(f"Error updating usage metrics: {e}")


    async def ainvoke(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Asynchronously invoke the EdenAI API with the given messages and tools.

        Args:
            messages: List of Message objects.

        Returns:
            Dict[str, Any]: The raw response from the EdenAI API.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "providers": [self.name],
            "text": self.format_messages(messages),
        }

        if self.tools:
            function_tools = self.convert_tools_to_api_format(self.tools)
            payload["available_tools"] = function_tools

        metrics = Metrics()
        metrics.response_timer.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    metrics.response_timer.stop()
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"EdenAI Error: {error_text}")
                        raise ValueError(f"EdenAI API Error: {error_text}")
                    
                    raw_response = await response.json()
                    return raw_response
        except Exception as e:
            metrics.response_timer.stop()
            logger.error(f"Error invoking EdenAI API asynchronously: {e}")
            raise

    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        """
        Asynchronously send a generate content request to EdenAI and return the response.

        Args:
            messages (List[Message]): The list of messages to send to the model.

        Returns:
            ModelResponse: The model response.
        """
        logger.debug("---------- EdenAI Async Response Start ----------")
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()
        metrics.response_timer.start()

        try:
            raw_response = await self.ainvoke(messages=messages)
            raw_response = raw_response[self.name]
            metrics.response_timer.stop()
        except Exception as e:
            metrics.response_timer.stop()
            logger.error(f"Error in EdenAI async response: {e}")
            raise

        response_message = None
        for msg in raw_response["message"]:
            if msg["role"] == "assistant":
                response_message = EdenaiChatMessage(
                    generated_text=msg.get("message", ""),
                    tools=msg.get("tools", ""),
                    tool_calls=msg.get("tool_calls", []),
                )

        if response_message is None:
            raise ValueError("No assistant response found in the EdenAI async response.")

        response_usage = {
            "input_token": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        assistant_message = self.create_assistant_message(
            response_message=response_message, metrics=metrics, response_usage=response_usage
        )

        messages.append(assistant_message)

        assistant_message.log()
        metrics.log()

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.audio is not None:
            model_response.audio = assistant_message.audio

        tool_role = "tool"
        if (
            self.handle_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                model_response=model_response,
                tool_role=tool_role,
            )
            is not None
        ):
            return self.handle_post_tool_call_messages(messages=messages, model_response=model_response)

        logger.debug("---------- EdenAI Async Response End ----------")
        return model_response
    


    def invoke_stream(self, messages: List[Message]) -> Iterator[dict]:
        """
        Invoke the EdenAI API with streaming enabled and parse JSON line responses.

        Args:
            messages: List of Message objects.

        Returns:
            Iterator[dict]: An iterator of parsed JSON objects from the response.
        """
        url = self.base_url + "/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "providers": [self.name],
            "text": self.format_messages(messages),
        }

        try:
            response = requests.post(url=url, headers=headers, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON line: {line}")
        except Exception as e:
            logger.error(f"Error during streaming API request: {e}")
            raise


    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        """
        Generate a streaming response from EdenAI.

        Args:
            messages (List[Message]): A list of messages.

        Returns:
            Iterator[ModelResponse]: An iterator of model responses.
        """
        logger.debug("---------- EdenAI Response Start ----------")
        self._log_messages(messages)
        metrics: Metrics = Metrics()

        metrics.response_timer.start()
        for response_chunk in self.invoke_stream(messages=messages):
            if "text" in response_chunk and not response_chunk.get("blocked", False):

                response_content = response_chunk["text"]
                yield ModelResponse(content=response_content)

        metrics.response_timer.stop()

        assistant_message = Message(role="assistant")
        if response_content != "":
            assistant_message.content = response_content

        messages.append(assistant_message)

        assistant_message.log()
        metrics.log()
        logger.debug("---------- EdenAI Response End ----------")


    async def ainvoke_stream(self, messages: List[Message]) -> AsyncIterator[dict]:
        """
        Asynchronously invoke the EdenAI API with streaming enabled and parse JSON line responses.

        Args:
            messages: List of Message objects.

        Returns:
            AsyncIterator[dict]: An async iterator of parsed JSON objects from the response.
        """
        url = self.base_url + "/stream"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "providers": [self.name],
            "text": self.format_messages(messages),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    buffer = "" 
                    async for line in response.content.iter_any():
                        if line:
                            try:
                                decoded_line = line.decode("utf-8")
                                buffer += decoded_line 
                                
                                while '\n' in buffer:  
                                    json_chunk, buffer = buffer.split('\n', 1)
                                    try:
                                        yield json.loads(json_chunk)
                                    except json.JSONDecodeError as e:
                                        logger.error(f"Failed to decode JSON line: {json_chunk}")
                            except Exception as e:
                                logger.error(f"Error while processing chunk: {e}")
        except Exception as e:
            logger.error(f"Error during streaming API request: {e}")
            raise

    async def aresponse_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """
        Generate a streaming response from EdenAI.

        Args:
            messages (List[Message]): A list of messages.

        Returns:
            AsyncIterator[ModelResponse]: An async iterator of model responses.
        """
        logger.debug("---------- EdenAI Response Start ----------")
        self._log_messages(messages)
        metrics = Metrics()

        metrics.response_timer.start()
        async for response_chunk in self.ainvoke_stream(messages=messages): 
            if "text" in response_chunk and not response_chunk.get("blocked", False):
                response_content = response_chunk["text"]
                yield ModelResponse(content=response_content)

        metrics.response_timer.stop()

        assistant_message = Message(role="assistant")
        if response_content != "":
            assistant_message.content = response_content

        messages.append(assistant_message)

        assistant_message.log()
        metrics.log()
        logger.debug("---------- EdenAI Response End ----------")