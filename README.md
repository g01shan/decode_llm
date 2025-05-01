# Can I Decode? 🤔

## Problem Definition
Ok, this is nothing fancy! It's just practice.

We are going to fetch a huggingface model and implement different decoding algorithms.
We want to compare my decoded output versus what huggingface does and THEY SHOULD MATCH!

Note that we want to do everything in batches. This requires attention to details that
are not relevant if you decoding a single string.

These are different algorithms implemented here:
- Greedy (see greedy.py)

_Note: Everything runs on CPU here! We are GPU poor! Oh, actually I used MPS!_

## How to run this?
It's simple! For every algorithm, simply pass your prompt to the correct script:
`python3 greedy.py -p "Tehran is" "Sing to me one song for joy and"`

## I can decode! ✅ 

### Greedy decoding
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

### Top-K decoding
Ok, here is a sample output:
```
- We are running this on mps!

Top-k decoding results from HuggingFace -- Completed in 0.83 seconds
+---------------------------------+-------------------------------------------------------------------------+
| Tehran is                       | Tehran is the world's first country to adopt a nuclear-resistance       |
+---------------------------------+-------------------------------------------------------------------------+
| Sing to me one song for joy and | Sing to me one song for joy and love, for love, for joy of all men, and |
+---------------------------------+-------------------------------------------------------------------------+

Top-k decoding results from us (with KV cache) -- Completed in 0.32 seconds
+---------------------------------+-------------------------------------------------------------------------+
| Tehran is                       | Tehran is the world's first country to adopt a nuclear-resistance       |
+---------------------------------+-------------------------------------------------------------------------+
| Sing to me one song for joy and | Sing to me one song for joy and love, for love, for joy of all men, and |
+---------------------------------+-------------------------------------------------------------------------+
```

Again, we have a perfect match!  
BTW, you might have noticed that we no longer have the solution without KV cache because why should we!

## Anything else? Any learnings?
* When passing KV cache to a HuggingFace model, it gets a bit tricky as what you exactly
need to pass:
    * **model_input**: Not so surprisingly, you only need to pass the token_ids of 
    whatever is not part of your KV. Basically, it often is only your new token.
    * **attention_mask**: You have to provide the attention mask for the entire sequence
    which includes what is in your KV and the new tokens you are processing. 
    * **position_ids**: You have to provide the position ids of the new tokens (and not
    what's in the KV).
    * All of these make sense if you think about it, but definitely can be confusing
    when you first try it.
* An algorithm like Top-K involves sampling, so you have to be careful to make sure you
get the same output as HuggingFace.
    * My initial implementation didn't match HF's result. Looking into it deeper, I 
    noticed that I've first reduced my space to K values and then took the sample, while
    HF set the weight of all other tokens to 0 and took a sample. The distribution is
    the same but multinomial returns different results.