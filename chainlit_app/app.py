import chainlit as cl


@cl.on_message
def main(message: cl.Message):
    result = message.content.title()  # Assuming 'content' is where the text is stored
    cl.send_message(content=f"Sure, here is the message: {result}")
