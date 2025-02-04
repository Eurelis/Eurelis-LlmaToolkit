from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper


def test_run_with_permanent_filters(
    chatbot_project1: ChatbotWrapper, chatbot_project2: ChatbotWrapper
):
    """
    Test the run method to ensure permanent filters integration functionality.
    """
    user_message = "Tell me about projects"
    response_project1 = chatbot_project1.run(message=user_message)
    response_project2 = chatbot_project2.run(message=user_message)

    assert response_project1 is not None, "The chatbot response should not be empty."
    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response_project1.source_nodes
    ), "All nodes should correspond to the project."

    assert response_project2 is not None, "The chatbot response should not be empty."
    assert all(
        node.node.metadata.get("project") == "eurelis2"
        for node in response_project2.source_nodes
    ), "All nodes should correspond to the project."


def test_run_with_project_filter(
    chatbot_project1: ChatbotWrapper, chatbot_project2: ChatbotWrapper
):
    """
    Test the run method to ensure project filter functionality.
    """
    user_message = "Tell me about projects"

    response_project1 = chatbot_project1.run(message=user_message)
    response_project2 = chatbot_project2.run(message=user_message)

    assert (
        response_project1 is not None
    ), "Response from eurelis1 chatbot should not be None."
    assert (
        response_project2 is not None
    ), "Response from eurelis2 chatbot should not be None."

    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response_project1.source_nodes
    ), "All nodes should belong to eurelis1."

    assert all(
        node.node.metadata.get("project") == "eurelis2"
        for node in response_project2.source_nodes
    ), "All nodes should belong to eurelis2."


def test_run_with_common_metadata(
    chatbot_project1_with_common_filter: ChatbotWrapper,
    chatbot_project2_with_common_filter: ChatbotWrapper,
):
    """
    Test the run method to ensure common metadata filters integration functionality.
    """
    user_message = "Tell me about your projects"

    response_project1 = chatbot_project1_with_common_filter.run(message=user_message)
    response_project2 = chatbot_project2_with_common_filter.run(message=user_message)

    assert (
        response_project1 is not None
    ), "Response from project1 chatbot should not be None."
    assert (
        response_project2 is not None
    ), "Response from project2 chatbot should not be None."

    assert all(
        node.node.metadata.get("common_field") == "common_value"
        for node in response_project1.source_nodes
    ), "All nodes should have the common metadata field for project1."
    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response_project1.source_nodes
    ), "All nodes should belong to eurelis1."

    assert all(
        node.node.metadata.get("common_field") == "common_value"
        for node in response_project2.source_nodes
    ), "All nodes should have the common metadata field for project2."
    assert all(
        node.node.metadata.get("project") == "eurelis2"
        for node in response_project2.source_nodes
    ), "All nodes should belong to eurelis2."


def test_run_with_specific_metadata(
    chatbot_project1: ChatbotWrapper,
    chatbot_project2: ChatbotWrapper,
    unique_filter_project1,
    unique_filter_project2,
):
    """
    Test the run method to ensure specific metadata filters integration functionality.
    """
    user_message = "Tell me about projects"

    response_project1 = chatbot_project1.run(
        message=user_message, filters=unique_filter_project1
    )
    response_project2 = chatbot_project2.run(
        message=user_message, filters=unique_filter_project2
    )

    assert (
        response_project1 is not None
    ), "Response from project1 chatbot should not be None."
    assert (
        response_project2 is not None
    ), "Response from project2 chatbot should not be None."

    assert all(
        node.node.metadata.get("unique_field_project1") == "eurelis_unique_value1"
        for node in response_project1.source_nodes
    ), "All nodes should have the unique metadata field for project1."
    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response_project1.source_nodes
    ), "All nodes should belong to eurelis1."

    assert all(
        node.node.metadata.get("unique_field_project2") == "eurelis_unique_value2"
        for node in response_project2.source_nodes
    ), "All nodes should have the unique metadata field for project2."
    assert all(
        node.node.metadata.get("project") == "eurelis2"
        for node in response_project2.source_nodes
    ), "All nodes should belong to eurelis2."


def test_run_with_cross_project_filters(
    chatbot_project1: ChatbotWrapper,
    chatbot_project2: ChatbotWrapper,
    unique_filter_project1,
    unique_filter_project2,
):
    """
    Test the run method to ensure cross-project filters functionality.
    """
    user_message = "Tell me about projects"

    response_project1_with_project2_filter = chatbot_project1.run(
        message=user_message, filters=unique_filter_project2
    )
    response_project2_with_project1_filter = chatbot_project2.run(
        message=user_message, filters=unique_filter_project1
    )

    assert (
        response_project1_with_project2_filter is not None
    ), "Response from project1 chatbot with project2 filter should not be None."
    assert (
        response_project2_with_project1_filter is not None
    ), "Response from project2 chatbot with project1 filter should not be None."

    assert (
        len(response_project1_with_project2_filter.source_nodes) == 0
    ), "Project1 should not return nodes with project2 filter."
    assert (
        len(response_project2_with_project1_filter.source_nodes) == 0
    ), "Project2 should not return nodes with project1 filter."


def test_run_with_common_and_specific_filters(
    chatbot_project1_with_common_filter: ChatbotWrapper,
    chatbot_project2_with_common_filter: ChatbotWrapper,
    unique_filter_project1,
    unique_filter_project2,
):
    """
    Test the run method to ensure integration of both common and specific filters.
    """
    user_message = "Tell me about projects"

    response_project1 = chatbot_project1_with_common_filter.run(
        message=user_message, filters=unique_filter_project1
    )
    response_project2 = chatbot_project2_with_common_filter.run(
        message=user_message, filters=unique_filter_project2
    )

    assert (
        response_project1 is not None
    ), "Response from project1 chatbot should not be None."
    assert (
        response_project2 is not None
    ), "Response from project2 chatbot should not be None."

    assert all(
        node.node.metadata.get("common_field") == "common_value"
        for node in response_project1.source_nodes
    ), "All nodes should have the common metadata field for project1."
    assert all(
        node.node.metadata.get("unique_field_project1") == "eurelis_unique_value1"
        for node in response_project1.source_nodes
    ), "All nodes should have the unique metadata field for project1."
    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response_project1.source_nodes
    ), "All nodes should belong to eurelis1."

    assert all(
        node.node.metadata.get("common_field") == "common_value"
        for node in response_project2.source_nodes
    ), "All nodes should have the common metadata field for project2."
    assert all(
        node.node.metadata.get("unique_field_project2") == "eurelis_unique_value2"
        for node in response_project2.source_nodes
    ), "All nodes should have the unique metadata field for project2."
    assert all(
        node.node.metadata.get("project") == "eurelis2"
        for node in response_project2.source_nodes
    ), "All nodes should belong to eurelis2."


def test_run_with_combined_permanent_filters(
    chatbot_with_combined_permanent_filter: ChatbotWrapper,
):
    """
    Test the run method to ensure integration of combined common and specific filters in permanent filters.
    """
    user_message = "Tell me about projects"

    response = chatbot_with_combined_permanent_filter.run(message=user_message)

    assert response is not None, "Response from chatbot should not be None."

    assert all(
        node.node.metadata.get("common_field") == "common_value"
        for node in response.source_nodes
    ), "All nodes should have the common metadata field."
    assert all(
        node.node.metadata.get("unique_field_project1") == "eurelis_unique_value1"
        for node in response.source_nodes
    ), "All nodes should have the unique metadata field."
    assert all(
        node.node.metadata.get("project") == "eurelis1"
        for node in response.source_nodes
    ), "All nodes should belong to eurelis1."
