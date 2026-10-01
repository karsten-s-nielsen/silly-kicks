from silly_kicks.mcp import server


def test_server_exposes_expected_tools():
    assert server.tool_names() == {"check_orientation", "diagnose_provider", "validate_construct_validity"}
