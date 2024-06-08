PROMPT = '''
    Based on the following observations from the interview, assess the suitability of the candidate.
    Consider emotions and postures detected.

    Overall Suitability Score: Calculated Score out of 10 by assigning scores to emotions (happy:2, surprise:0, fear:-2)
    Overall Result: Suitable if candidate score is greater than 6,  Not suitable if candidate score is less than or equal to 6
    Reasoning: Explain how specific emotions and postures influenced the candidate's performance. Discuss strengths and areas for improvement.

    Generate output in the following format and do not use frame numbers in the analysis at all:

    Based on the observations from the interview, here is the assessment of the candidate's suitability:

    Candidate Assessment:

    Overall Suitability Score: X/10

    Overall Result:

    Reasoning:

    Strengths:

    Areas for Improvement:

    '''
