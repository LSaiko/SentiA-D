#!/usr/bin/env python3
"""Create the GitHub repo (if needed) and push the current branch.

Token is read from --token or $GITHUB_TOKEN — never stored. Usage:

    python scripts/push_to_github.py --token $GITHUB_TOKEN --user LSaiko --repo SentiA-D
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def _ensure_remote_repo(user: str, repo: str, token: str):
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=json.dumps({"name": repo, "private": False}).encode(),
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github+json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        print(f"Created {user}/{repo}")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            print(f"{user}/{repo} already exists — pushing to it")
        else:
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    ap.add_argument("--user", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--branch", default="main")
    args = ap.parse_args()
    if not args.token:
        sys.exit("No token: pass --token or set GITHUB_TOKEN")

    _ensure_remote_repo(args.user, args.repo, args.token)
    url = f"https://{args.token}@github.com/{args.user}/{args.repo}.git"
    subprocess.run(["git", "remote", "remove", "origin"], cwd=ROOT)  # ok if absent
    _git("remote", "add", "origin", url)
    _git("branch", "-M", args.branch)
    _git("push", "-u", "origin", args.branch)
    print("Pushed.")


if __name__ == "__main__":
    main()
