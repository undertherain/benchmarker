#!/usr/bin/env python3
import json
from system_query import system_query

if __name__ == "__main__":
    info = system_query.query_all()
    print(json.dumps(info, sort_keys=True, indent=4, separators=(',', ': ')))
