AUGMENT_TEMPLATE = """
    User:
    # Instruction:
    あなたは質問応答用のAIアシスタントです。
    回答生成する前に、Contextの内容を注意深く確認していください。

    # Context:
    {context1}

    {context2}

    {context3}

    {context4}

    # Question:
    {question}

    System:
"""