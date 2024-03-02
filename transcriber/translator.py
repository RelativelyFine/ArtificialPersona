def translate(from_language, to_language, text, client):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a sentence in {from_language} and your task is to translate it into {to_language}."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature = 0.7,
        top_p = 1
    )
    return response.choices[0].message.content