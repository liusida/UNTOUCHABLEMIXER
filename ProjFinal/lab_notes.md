# Apr 23 (1)

I am going to download the data from 2007 and 2017, and make them into one single variable data_X

Clean step for label Y:

1. manually change two extension filenames from lower case to upper case. ".xpt" -> ".XPT"

2. There are a few "Not known" and "Refuse", turn them to NaN (to del them in Step 3).

3. Drop NaN is 9 questions. note: DPQ100 is ok to be NaN.

4. Mark missing data in DPQ100 as 8, so that we can work in integers

And produce two response variables "Major depression" (0 or 1) and "PHQ score" (0-27)
