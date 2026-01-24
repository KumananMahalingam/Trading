from groq import Groq
import secret_key

# Then later, initialize the client
groq_client = Groq(api_key=secret_key.GROQ_API_KEY)

try:
    test_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✓ Groq API connection successful")
except Exception as e:
    print(f"✗ Groq API error: {e}")
    print("  Check that GROQ_API_KEY is set correctly in secret_key.py")