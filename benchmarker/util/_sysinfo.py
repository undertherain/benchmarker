#!/usr/bin/env python3
import json
from system_query import query_all

if __name__ == "__main__":
    info = query_all()
    print(json.dumps(info, sort_keys=True, indent=4, separators=(',', ': ')))
