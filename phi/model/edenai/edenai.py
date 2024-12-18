from time import time
import requests
import json
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Dict, Any

from phi.model.base import Model
from phi.model.message import Message
from phi.model.response import ModelResponse
from phi.utils.log import logger
from phi.utils.timer import Timer
from os import getenv


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


class EdenAI(Model):
    """
    Class for interacting with the EdenAI API.
    """

    id: str = "openai/gpt-4"  
    name: str = "EdenAI"
    provider: str = "EdenAI"

    api_key: str = getenv("EDENAI_API_KEY", None) 
    base_url: str = "https://api.edenai.run/v2/multimodal/chat"

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Format messages to match the EdenAI API structure.

        Args:
            messages: List of Message objects.

        Returns:
            List[Dict[str, Any]]: Formatted messages compatible with EdenAI's API.
        """
        formatted = []
        for msg in messages:
            content = [{"type": "text", "content": {"text": msg.content}}]
            formatted.append({"role": msg.role, "content": content})
        return formatted

    def invoke(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Invoke the EdenAI API with the given messages.

        Args:
            messages: List of Message objects.

        Returns:
            Dict[str, Any]: The raw response from the EdenAI API.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "providers": [self.id],  
            "messages": self.format_messages(messages),
        }

        logger.debug(f"Sending request to EdenAI with provider '{self.id}': {json.dumps(payload, indent=2)}")

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
            response: str,
            metrics: Metrics,
            response_cost: Optional[float] = None,
    ) -> Message:
        """
        Create an assistant message from the EdenAI response.
    
        Args:
            response: The text response (e.g., `generated_text`).
            metrics: Metrics object for usage tracking.
    
        Returns:
            Message object containing the assistant message.
        """
        usage_data = response_cost
        assistant_message = Message(role="assistant", content=response) 
        self.update_usage_metrics(assistant_message, metrics, usage_data)
    
        return assistant_message

    def update_usage_metrics(
        self, assistant_message: Message, metrics: Metrics, usage: Optional[Dict[str, Any]] = None
    ) -> None:
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

    def update_usage_metrics(
        self, assistant_message: Message, metrics: Metrics, usage: Optional[Dict[str, Any]] = None
    ) -> None:
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
            raw_response = self.invoke(messages=messages)  
            metrics.response_timer.stop()
        except Exception as e:
            logger.error(f"Error in EdenAI response: {e}")
            metrics.response_timer.stop()
            raise

        # parse response
        provider_response = raw_response.get(str(self.id), {}).get('generated_text', '')

        logger.debug(f"Assistant Response: {provider_response}")

        # create assistnat message
        assistant_message = self.create_assistant_message(
            response=provider_response,
            metrics=metrics
        )

        messages.append(assistant_message)

        
        assistant_message.log()
        metrics.log()


        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()

        logger.debug("---------- EdenAI Response End ----------")
        return model_response