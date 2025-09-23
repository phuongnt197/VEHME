
SYSTEM_PROMPT = """\
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."""


USER_PROMPT = """\
The following contains a math problem consisting of both a Question and a Student Answer. Your task is to analyze the Student Answer to determine whether there is any error. Begin by providing a brief reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer. After the reasoning, provide a final output indicating whether the Student Answer is correct or incorrect. Please follow the exact format below without adding any extra information:
<reasoning>
Brief reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer.
</reasoning>
<correctness>
Correct or Incorrect
</correctness>

Question: {question}
Student Answer: {student_answer}
"""

USER_PROMPT_DETAILED_AND_LOCALIZATION = """\
The following contains a math problem consisting of both a Question and a Student Answer. Your task is to analyze the Student Answer to determine whether there is any error. Begin by providing a detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer. After the reasoning, provide a final output indicating whether the Student Answer is correct or incorrect. Please follow the exact format below without adding any extra information:
<think>
Detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer.
</think>
<correctness>
Correct or Incorrect
</correctness>
<localization>
The localization of the error in the Student Answer. Write None if there is no error.
</localization>

Question: {question}
Student Answer: {student_answer}
"""

PATTERN = r"^<think>.*?</think>\s*<correctness>(.*?)</correctness>(?![\s\S])"

PATTERN_DETAILED_AND_LOCALIZATION = r"^<think>.*?</think>\s*<correctness>(.*?)</correctness>\s*<localization>(.*?)</localization>(?![\s\S])"