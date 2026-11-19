# VEYLITH-22 · Matched punctuation and word interventions

This protocol fixes the selection rule and support thresholds before scoring. Use the update-500 model, block 1, and the two frozen validation samples from the [expert 3 study](ROUTING-STUDY.md). Analyze the original and confirmation samples separately. Each contains 80 sequences from each of Cosmopedia v2, FineWeb-Edu, and Wikipedia. Each 512-token sequence contributes 511 valid input/next-token pairs.

## Select one pair per sequence

Use ordinary routing to find input positions where expert 3 is ranked first. Measure confidence as the weight of rank one after softmax over the top two ordinary gate logits. Exclude special SentencePiece pieces enclosed by `<` and `>`. Strip one leading `▁`. The punctuation class contains any piece with `.`, `!`, or `?`. The word class contains a nonempty piece consisting entirely of ASCII letters A–Z or a–z. A piece may represent part of a word.

Within a sequence, consider every punctuation/word pair within **32 input positions** and **0.05 confidence** of each other, inclusive. Compare the original float32 confidence values after promoting them to float64. Minimize `absolute position gap / 32 + absolute confidence gap / 0.05`. Break ties by smaller absolute position gap, then earlier punctuation index, then earlier word index. Keep at most one pair per sequence. Freeze all selections in a sample before reading next-token losses. Report eligible, selected and unselected counts for each class, sequence coverage, and position and confidence balance before and after matching, pooled and by source.

## Support and measurement

Require **at least 120 of 240 matched sequences overall and at least 30 of 80 in each source**, per sample. If either criterion fails, publish that sample's coverage and balance with no effect estimates. Keep these thresholds fixed for both samples.

For a supported sample, score ordinary routing and two separate interventions. One changes only the selected punctuation position, the other only the selected word position. In each, transfer expert 3's weight to the highest-scoring unused expert, retaining the second expert and its weight. Check that CE preceding the single replacement in each arm is exactly equal to ordinary CE, including all targets in rows without a selected pair.

The primary outcome is the mean over matched sequences of **punctuation CE increase minus word CE increase**, in nats. Each increase uses the next-token loss at its own selected position relative to ordinary routing. Report both arm means as well. Bootstrap entire sequences, sampling 80 with replacement within each of the three sources. Use the frozen PCG64 stream (seed 20261118), 1,000 draws and linear 2.5th and 97.5th percentiles. Each replicate retains both measurements for a matched sequence and omits unmatched sequences from the mean. Report the range of matched counts across replicates and the number with a defined mean.

A positive contrast favors a larger effect of replacing expert 3 at matched punctuation pieces; a negative contrast favors a larger effect at matched word pieces. Assess the direction on each sample, with replication requiring adequate coverage and both intervals strictly on the same side of zero. The estimate applies only to matched positions at this checkpoint.
