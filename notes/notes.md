**Graph-based bot detection** treats a social network not as a list of accounts, but as a relational structure — the insight being that bots behave differently *in relation to others*, not just in isolation. Three pillars underpin this:

**1. The relational assumption.** Bots tend to cluster: they follow each other, retweet in coordinated bursts, and form suspiciously dense subgraphs. Genuine users form sparse, organic communities with diverse connectivity. This makes the graph topology itself a detection signal.

**2. Homophily exploitation.** In social networks, similar nodes tend to connect. Bot farms are built cheaply, so their members share behavioral fingerprints — and those fingerprints propagate through edges. Label propagation and graph neural networks both exploit this.

**3. Feature duality.** Every node carries two signal types — its own attributes (followers count, tweet frequency, account age) and its structural position (degree, clustering coefficient, PageRank). The best detectors fuse both.

The standard pipeline has four stages: construct the graph, extract features (node + structural), train a classifier (ranging from classical ML to full GNNs), then evaluate. Here's how that looks:Click any box to drill into that concept. Now here's how cresci-2017 maps onto this framework specifically.

**Your dataset is actually an ideal testbed** because it captures the full bot taxonomy: fake followers (inflated social metrics, no real activity), social spambots (coordinate around trending content), traditional spambots (classic link-spam patterns), and genuine accounts as the negative class. Each category stresses the model differently — fake followers are hard to catch on profile features alone because they look plausible, whereas traditional spambots are nearly trivially caught by tweet content.

**The standard baseline approach for cresci-2017** goes like this:

1. **Build a follower/retweet graph** — merge `users.csv` across all categories as nodes, assign binary labels (bot vs genuine), then construct edges from the `tweets.csv` files (retweet relationships are parseable from tweet metadata).
2. **Extract node features** from `users.csv` — the standard fields are `followers_count`, `friends_count`, `statuses_count`, `listed_count`, `favourites_count`, account age, `verified`, and profile completeness indicators.
3. **Run a Random Forest baseline** — this is the standard first checkpoint in the literature. It uses only node features, no graph structure. This gives you a feature-only ceiling to beat.
4. **Add graph structure** — compute degree, clustering coefficient, and PageRank per node, append to feature matrix, re-run. The delta tells you how much the graph adds.
5. **GNN tier** — fit a GCN or GraphSAGE on the same graph. This is where neighborhood aggregation should catch the coordinated spambot clusters that isolated features miss.

The known challenge with this dataset is **class imbalance and category leakage** — the bot classes aren't evenly sized and some bot types are easier than others, so aggregated F1 can be misleading. The literature standard is to report per-category F1 in addition to macro averages.
