# Can I Decode? 🤔

### Problem Definition
Ok, this is nothing fancy! It's just practice.

We are going to fetch a huggingface model and implement different decoding algorithms.
We want to compare my decoded output versus what huggingface does and THEY SHOULD MATCH!

Note that we want to do everything in batches. This requires attention to details that
are not relevant if you decoding a single string.

These are different algorithms implemented here:
- Greedy (see greedy.py)

_Note: Everything runs on CPU here! We are GPU poor! Oh, actually I used MPS!_

### How to run this?
It's simple! For every algorithm, simply pass your prompt to the correct script:
`python3 greedy.py -p "Tehran is" "Sing to me one song for joy and"`

### I can decode! ✅ 

#### Greedy decoding
Ok, here is a sample output:
```
- We are running this on mps!

Greedy decoding results from HuggingFace -- Completed in 1.10 seconds
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Paris is                        | Paris is a city of people, of people, of people. It's a place where people come together to make a |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Sing to me one song for joy and | Sing to me one song for joy and I'm going to sing it to you.                                       |
|                                 |                                                                                                    |
|                                 | I'm going to sing it to you.                                                                       |
+---------------------------------+----------------------------------------------------------------------------------------------------+

Greedy decoding results from us (with KVout cache) -- Completed in 1.83 seconds
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Paris is                        | Paris is a city of people, of people, of people. It's a place where people come together to make a |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Sing to me one song for joy and | Sing to me one song for joy and I'm going to sing it to you.                                       |
|                                 |                                                                                                    |
|                                 | I'm going to sing it to you.                                                                       |
+---------------------------------+----------------------------------------------------------------------------------------------------+

Greedy decoding results from us (with KV cache) -- Completed in 0.58 seconds
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Paris is                        | Paris is a city of people, of people, of people. It's a place where people come together to make a |
+---------------------------------+----------------------------------------------------------------------------------------------------+
| Sing to me one song for joy and | Sing to me one song for joy and I'm going to sing it to you.                                       |
|                                 |                                                                                                    |
|                                 | I'm going to sing it to you.                                                                       |
+---------------------------------+----------------------------------------------------------------------------------------------------+
```

And as you can see our implementation (with & without KV cache) matches HuggingFace.
Interestingly, our solution is faster. But to be fair, we are using the beam search 
implementation of HF with beam_size = 1. Probably this is some unnecessary bookkeeping.

### Anything else? Any learnings?
\<\<To be written ...\>\>