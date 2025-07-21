import json

from enthusiast_agent_re_act import BaseReActAgent

class DataExtractionReActAgent(BaseReActAgent):
    def get_answer(self, input_text: str) -> str:
        agent_executor = self._build_agent_executor()
        template_dict = {
            "check_in_time": "<time>",
            "checkout_time": "<time>",
            "pets_allowed": "<boolean>",
            "quite_hours": "<time-time>",
            "address": "<full address string>",
            "price": "<number with currency>",
            "rating": "<number>"
        }

        output_format = json.dumps(template_dict)
        agent_output = agent_executor.invoke(
            {
                "input": input_text,
                "output_format": output_format,
                "products_type": "Hotel",
            },
            config=self._build_invoke_config()
        )
        result = agent_output["output"]
        try:
            parsed = json.loads(result)
            result = json.dumps(parsed, indent=4)
        except json.JSONDecodeError:
            pass

        return result
