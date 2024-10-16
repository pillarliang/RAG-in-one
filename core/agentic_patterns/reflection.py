from model.llm import LLM


class ReflectionAgent:
    def __init__(self):
        self.client = LLM()

    def generate(self, generation_history: list) -> str:
        """
        Generates a response based on the provided generation history.

        Args:
            generation_history (list): List of messages forming the conversation history

        """
        response = self.client.get_response(generation_history)

        return response

    def reflect(self, reflection_history: list) -> str:
        """
        Reflects on the generation by generating a critique or feedback.

        Args:
            reflection_history (list): List of messages forming the reflection history.
        """
        response = self.client.get_response(reflection_history)

        return response

    def run(self, generation_system_prompt: str, reflection_system_prompt: str, user_prompt: str, n_steps: int = 2) -> str:
        generation_history = [
            {"role": "system", "content": generation_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        reflection_history = [
            {"role": "system", "content": reflection_system_prompt},
        ]

        generation = ""

        for step in range(n_steps):
            # Generate the output based on generation history
            generation = self.generate(generation_history)

            # Update histories
            generation_history.append(
                {"role": "assistant", "content": generation}
            )
            reflection_history.append(
                {"role": "assistant", "content": generation}
            )

            # Reflect and critique the generation
            critique = self.reflect(reflection_history)
            reflection_history.append(
                {"role": "assistant", "content": critique}
            )
            generation_history.append(
                {"role": "user", "content": critique}
            )

        return generation


if __name__ == "__main__":
    generation_system_prompt = """
    You are a Python programmer tasked with generating high quality Python code.
    Your task is to Generate the best content possible for the user's request. If the user provides critique,
    respond with a revised version of your previous attempt."""

    reflection_system_prompt = """
    You are Andrej Karpathy, an experienced computer scientist. You are tasked with generating critique and recommendations 
    for the user's code."""

    user_prompt = """
    Generate a Python implementation of the Merge Sort algorithm"""
    agent = ReflectionAgent()

    final_response = agent.run(
        generation_system_prompt=generation_system_prompt,
        reflection_system_prompt=reflection_system_prompt,
        user_prompt=user_prompt,
        n_steps=3,
    )
