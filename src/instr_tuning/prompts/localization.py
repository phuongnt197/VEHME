SYSTEM_PROMPT = """\
You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."""



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

Question: <image>
Reference Answer: {reference_answer}
Student Answer: <image>
"""

USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL = """\
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

"""

FERMAT_USER_PROMPT_DETAILED_AND_LOCALIZATION = """\
The following is a single image containing both a Question and a Student Answer. Your task is to analyze the Student Answer to determine whether there is any error. Begin by providing a detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer. After the reasoning, provide a final output indicating whether the Student Answer is correct or incorrect. The error location must be a specific mathematical expression in the Student Answer. Please follow the exact format below without adding any extra information:
<think>
Detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer.
</think>
<correctness>
Correct or Incorrect
</correctness>
<localization>
The localization of the error in the Student Answer. Write None if there is no error.
</localization>

Question and Student Answer: <image>
Reference Answer: {reference_answer}
"""

FERMAT_USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL = """\
The following is a single image containing both a Question and a Student Answer. Your task is to analyze the Student Answer to determine whether there is any error. Begin by providing a detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer. After the reasoning, provide a final output indicating whether the Student Answer is correct or incorrect. The error location must be a specific mathematical expression in the Student Answer. Please follow the exact format below without adding any extra information:
<think>
Detailed reasoning for your analysis, explaining where and why you believe an error is present or absent in the Student Answer.
</think>
<correctness>
Correct or Incorrect
</correctness>
<localization>
The localization of the error in the Student Answer. Write None if there is no error.
</localization>

[EXAMPLE-START]
<think>\nThe Student Answer correctly identifies the slope and y-intercept of the original line as 2 and 3, respectively. The equation of the translated line is also correctly written as \\( y = 2x + 3 + a \\). The substitution of the point (5, 6) into the equation is also done correctly: \\( 6 = 2 \\times 5 + 3 + a \\), which simplifies to \\( 6 = 13 + a \\). The final step of solving for \\( a \\) is correct as well: \\( a = -7 \\). The Student Answer follows the same logical steps as the Reference Answer, and no errors are present in the calculations or reasoning.\n\n</think>\n<correctness>\nCorrect\n</correctness>\n<localization>\nNone\n</localization>
[EXAMPLE-END]
"""

PATTERN = r"^<think>.*?</think>\s*<correctness>(.*?)</correctness>(?![\s\S])"

PATTERN_DETAILED_AND_LOCALIZATION = r"<correctness>(.*?)</correctness>\s*<localization>(.*?)</localization>(?![\s\S])"