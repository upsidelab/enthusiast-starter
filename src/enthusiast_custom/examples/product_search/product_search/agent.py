from enthusiast_agent_re_act import BaseReActAgent


class ProductSearchReActAgent(BaseReActAgent):
    def get_answer(self, input_text: str) -> str:
        agent_executor = self._build_agent_executor()
        agent_output = agent_executor.invoke(
            {"input": input_text, "products_type": "any"},
            config=self._build_invoke_config(),
        )
        return agent_output["output"]
