from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
)


def test_initialize_memory_with_history(chatbot, memory_persistence):
    """
    Test to ensure _initialize_memory correctly initializes memory and loads conversation history.
    """
    memory_persistence.load_history()

    # Call _initialize_memory
    initialized_memory = chatbot._initialize_memory("test_conversation")

    assert initialized_memory is not None, "Initialized memory should not be None"
    assert (
        initialized_memory == memory_persistence._memory
    ), "Loaded memory does not match expected memory"


def test_memory_persistence_creation(chatbot, memory):
    """
    Test to ensure _get_memory_persistence correctly creates a persistence instance.
    """
    persistence = chatbot._get_memory_persistence(memory)

    assert persistence is not None, "Memory persistence should not be None"
    assert (
        persistence._memory == memory
    ), "Memory in persistence does not match provided memory"


def test_memory_persistence_reuse(chatbot, memory):
    """
    Test to ensure _get_memory_persistence reuses an existing persistence instance.
    """
    # Create initial persistence
    persistence = chatbot._get_memory_persistence(memory)

    # Retrieve existing persistence
    reused_persistence = chatbot._get_memory_persistence()

    assert (
        reused_persistence == persistence
    ), "Memory persistence instance should be reused"


def test_run_with_memory_and_persistence(chatbot, memory_config):
    """
    Test the run method to ensure memory integration and persistence functionality.
    """
    # Simuler une nouvelle interaction via run
    user_message = "How are you today?"
    conversation_id = "test_conversation"
    response = chatbot.run(user_message)

    # Step 1: Vérifier que la réponse du chatbot est non vide
    assert response is not None, "La réponse du chatbot ne doit pas être vide."
    assert isinstance(
        response.response, str
    ), "La réponse doit être une chaîne de caractères."

    # Step 2: Vérifier que le message utilisateur a été ajouté à la mémoire
    updated_messages = (
        chatbot._get_memory_persistence()._memory.chat_store.get_messages(
            conversation_id
        )
    )
    assert (
        updated_messages[-2].content == user_message
    ), "L'avant dernier message de la conversation devrait être le message utilisateur."
    assert (
        updated_messages[-1].content == response.response
    ), "Le dernier message de la conversation devrait être la réponse du chatbot."

    # Step 3: Recharger une mémoire vide à partir de la même config pour vérifier si la conversation a été sauvegardé
    reloaded_memory = MemoryFactory.create_memory(memory_config, "test_conversation")
    reloaded_persistence = MemoryPersistenceFactory.create_memory_persistence(
        chatbot._get_memory_persistence()._config, reloaded_memory, conversation_id
    )
    reloaded_persistence.load_history()

    # Vérifier que les données rechargées incluent les nouveaux messages
    reloaded_messages = reloaded_persistence._memory.chat_store.get_messages(
        conversation_id
    )
    assert (
        reloaded_messages[-2].content == user_message
    ), "L'avant dernier message de la conversation rechargée devrait être le message utilisateur."
    assert (
        reloaded_messages[-1].content == response.response
    ), "Le dernier message de la conversation rechargée devrait être la réponse du chatbot."
