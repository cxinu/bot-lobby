#!/usr/bin/env bash

# 1. Column headers for all users.csv files
for f in data/*/users.csv; do
  echo "=== $f ==="; head -1 "$f"; done

# 2. Column headers for tweet files (only some dirs have them)
for f in data/*/tweets.csv; do
  echo "=== $f ==="; head -1 "$f"; done

# 3. Row counts per file
echo "--- ROW COUNTS ---"
for f in data/**/*.csv; do
  echo "$(wc -l < "$f") $f"; done

# 4. Spot-check a genuine user row and a bot row
echo "--- genuine sample ---"
sed -n '2p' data/genuine_accounts/users.csv

echo "--- fake_followers sample ---"
sed -n '2p' data/fake_followers/users.csv

echo "--- social_spambot_1 sample ---"
sed -n '2p' data/social_spambots_1/users.csv
