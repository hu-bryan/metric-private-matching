"""Verify immutable scientific assets without importing numerical libraries."""
import json
from common import verify_bundle

if __name__ == '__main__':
    print(json.dumps(verify_bundle(), indent=2))
