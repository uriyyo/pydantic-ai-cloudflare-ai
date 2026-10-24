from pydantic_ai_cloudflare_ai.headers import CommonHeaders, Header


def test_header_sets_are_plain_mappings():
    assert CommonHeaders.JSON == {"content-type": "application/json"}
    assert dict(CommonHeaders.JSON) == {"content-type": "application/json"}
    assert Header.GATEWAY_AUTHORIZATION.bearer("t") == {"cf-aig-authorization": "Bearer t"}
