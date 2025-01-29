from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper


def test_run_with_filters(chatbot_config, filters):
    """
    Test the run method to ensure filters integration functionality.
    """
    chatbot = ChatbotWrapper(chatbot_config, "default_test_u", filters)

    # Simuler une nouvelle interaction via run
    user_message = "How are you today?"
    response = chatbot.run(message=user_message)

    assert response is not None, "La réponse du chatbot ne doit pas être vide."

    # TODO : Vérifier les sources
    print(response.sources)
