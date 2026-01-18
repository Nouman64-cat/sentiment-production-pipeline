import pytest
from src.preprocessing.clean import clean_text

def test_clean_text_basic():
    """Test that basic string cleaning works (lowercase + strip)."""
    raw = "  Hello WORLD  "
    expected = "hello world"
    assert clean_text(raw) == expected

def test_clean_text_removes_html():
    """Test removal/unescaping of HTML entities."""
    raw = "I loved it &lt;3"
    expected = "i loved it <3"
    # Note: Your current cleaner unescapes HTML but keeps the '<' 
    # unless strictly removed by regex. Let's verify behavior.
    assert clean_text(raw) == expected

def test_clean_text_removes_urls():
    """Test removal of http/https/www links."""
    raw = "Check this https://google.com and www.example.com now"
    expected = "check this and now"
    assert clean_text(raw) == expected

def test_clean_text_removes_handles():
    """Test removal of twitter handles (@username)."""
    raw = "Hey @elonmusk look at this"
    expected = "hey look at this"
    assert clean_text(raw) == expected

def test_clean_text_removes_special_chars():
    """Test removal of non-alphanumeric characters except punctuation."""
    # Your regex allows [^a-zA-Z0-9\s.,!?]
    raw = "Price is $50.00!!"
    expected = "price is 50.00!!"
    assert clean_text(raw) == expected

def test_clean_text_handles_empty_input():
    """Test edge cases like None or empty strings."""
    assert clean_text("") == ""
    assert clean_text(None) == ""