from google import genai

client = genai.Client(api_key="")

question = "The proper divisors of 12 are 1, 2, 3, 4 and 6. A proper divisor of an integer $N$ is a positive divisor of $N$ that is less than $N$. What is the sum of the proper divisors of the sum of the proper divisors of 284?"
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-05-20", contents=question
)
print(response.text)