import pytest
from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper


@pytest.mark.parametrize(
    "chatbot_project, project_name",
    [
        (("chatbot_wrapper/test_chatbot_config_project1.json", "eurelis1"), "eurelis1"),
        (("chatbot_wrapper/test_chatbot_config_project2.json", "eurelis2"), "eurelis2"),
    ],
    indirect=["chatbot_project"],  # Indique que chatbot_project est une fixture
)
def test_run_with_permanent_metadata_filters(
    chatbot_project: ChatbotWrapper, project_name: str
):
    """
    Test the run method to ensure permanent filters integration functionality.
    """
    user_message = "Tell me about projects"
    response = chatbot_project.run(message=user_message)

    assert response is not None, "The chatbot response should not be empty."
    assert all(
        source_node.node.metadata.get("project") == project_name
        for source_node in response.source_nodes
    ), f"All nodes should correspond to the project {project_name}."


@pytest.mark.parametrize(
    "chatbot_project_with_permanent_filter, project_name",
    [
        (("chatbot_wrapper/test_chatbot_config_project1.json", "eurelis1"), "eurelis1"),
        (("chatbot_wrapper/test_chatbot_config_project2.json", "eurelis2"), "eurelis2"),
    ],
    indirect=["chatbot_project_with_permanent_filter"],
)
def test_run_with_common_metadata_project1(
    chatbot_project_with_permanent_filter: ChatbotWrapper, project_name: str
):
    """
    Test the run method to ensure common metadata filters integration functionality for project.
    """
    user_message = "Tell me about your projects"
    response_project = chatbot_project_with_permanent_filter.run(message=user_message)

    assert (
        response_project is not None
    ), f"Response from {project_name} chatbot should not be None."
    assert all(
        source_node.node.metadata.get("common_field") == "common_value"
        for source_node in response_project.source_nodes
    ), f"All nodes should have the common metadata field for {project_name}."
    assert all(
        source_node.node.metadata.get("project") == project_name
        for source_node in response_project.source_nodes
    ), f"All nodes should belong to {project_name}."


@pytest.mark.parametrize(
    "chatbot_project, metadata_filters, project_name, unique_field, unique_value",
    [
        (
            ("chatbot_wrapper/test_chatbot_config_project1.json", "eurelis1"),
            {"unique_field_project1": "eurelis_unique_value1"},
            "eurelis1",
            "unique_field_project1",
            "eurelis_unique_value1",
        ),
        (
            ("chatbot_wrapper/test_chatbot_config_project2.json", "eurelis2"),
            {"unique_field_project2": "eurelis_unique_value2"},
            "eurelis2",
            "unique_field_project2",
            "eurelis_unique_value2",
        ),
    ],
    indirect=["chatbot_project", "metadata_filters"],
)
def test_run_with_specific_metadata(
    chatbot_project: ChatbotWrapper,
    metadata_filters,
    project_name,
    unique_field,
    unique_value,
):
    """
    Test the run method to ensure specific metadata filters integration functionality dynamically.
    """
    user_message = "Tell me about projects"
    response = chatbot_project.run(
        message=user_message, metadata_filters=metadata_filters
    )

    assert response is not None, "Response from chatbot should not be None."
    assert all(
        source_node.node.metadata.get(unique_field) == unique_value
        for source_node in response.source_nodes
    ), f"All nodes should have the unique metadata field {unique_field} with value {unique_value}."

    assert all(
        source_node.node.metadata.get("project") == project_name
        for source_node in response.source_nodes
    ), f"All nodes should belong to {project_name}."


@pytest.mark.parametrize(
    "chatbot_project, metadata_filters, project_name, cross_project_name",
    [
        (
            ("chatbot_wrapper/test_chatbot_config_project1.json", "eurelis1"),
            {"project": "eurelis2"},
            "eurelis1",
            "eurelis2",
        ),
        (
            ("chatbot_wrapper/test_chatbot_config_project2.json", "eurelis2"),
            {"project": "eurelis1"},
            "eurelis2",
            "eurelis1",
        ),
    ],
    indirect=["chatbot_project", "metadata_filters"],
)
def test_run_with_cross_project_filters(
    chatbot_project: ChatbotWrapper, metadata_filters, project_name, cross_project_name
):
    """
    Test the run method to ensure cross-project filters functionality dynamically.
    """
    user_message = "Tell me about projects"
    response = chatbot_project.run(
        message=user_message, metadata_filters=metadata_filters
    )

    assert (
        response is not None
    ), f"Response from chatbot ({project_name}) with cross-project filter ({cross_project_name}) should not be None."

    assert (
        len(response.source_nodes) == 0
    ), f"Chatbot for {project_name} should not return nodes from {cross_project_name}."


@pytest.mark.parametrize(
    "chatbot_project, metadata_filters, project_name, common_field, common_value",
    [
        (
            ("chatbot_wrapper/test_chatbot_config_project1.json", "eurelis1"),
            {
                "common_field": "common_value",
                "unique_field_project1": "eurelis_unique_value1",
            },
            "eurelis1",
            "common_field",
            "common_value",
        ),
        (
            ("chatbot_wrapper/test_chatbot_config_project2.json", "eurelis2"),
            {
                "common_field": "common_value",
                "unique_field_project2": "eurelis_unique_value2",
            },
            "eurelis2",
            "common_field",
            "common_value",
        ),
    ],
    indirect=["chatbot_project", "metadata_filters"],
)
def test_run_with_common_and_specific_filters(
    chatbot_project: ChatbotWrapper,
    metadata_filters,
    project_name,
    common_field,
    common_value,
):
    """
    Test the run method to ensure integration of both common and specific filters dynamically.
    """
    user_message = "Tell me about projects"

    response = chatbot_project.run(
        message=user_message, metadata_filters=metadata_filters
    )

    assert (
        response is not None
    ), f"Response from chatbot ({project_name}) should not be None."

    assert all(
        source_node.node.metadata.get(common_field) == common_value
        for source_node in response.source_nodes
    ), f"All nodes should have the common metadata field `{common_field}={common_value}` for {project_name}."

    # Vérifier que toutes les clés et valeurs de metadata_filters sont présentes
    for metadata_filter in metadata_filters.filters:
        assert all(
            source_node.node.metadata.get(metadata_filter.key) == metadata_filter.value
            for source_node in response.source_nodes
        ), f"All nodes should have the metadata field `{metadata_filter.key}={metadata_filter.value}` for {project_name}."

    assert all(
        source_node.node.metadata.get("project") == project_name
        for source_node in response.source_nodes
    ), f"All nodes should belong to {project_name}."
