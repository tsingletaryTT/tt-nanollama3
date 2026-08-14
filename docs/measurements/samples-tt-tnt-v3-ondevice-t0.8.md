<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->

# Samples — tt-tnt-v3, ON DEVICE, sampled

model: `episod/tt-tnt` (tt-tnt-v3 weights) served through the Tenstorrent vLLM plugin on one
Blackhole p300c chip, mesh (1,1) · sampling (temperature=0.8, top_p=0.95) · seeds 0 and 1 ·
60 new tokens · 2 samples/prompt · frozen 15-prompt evaluation set.

## Why this exists

Every on-device decode measurement taken on this project before today was **greedy**, and
greedy decoding manufactures repetition attractors in a 22M-parameter model — so the recorded
free-running result (median agreement with CPU of 3–4 tokens out of 40) risked overstating the
defect. This file is the control: the same served model, sampled, read on its own quality
terms rather than against a CPU token stream it cannot match by construction.

The CPU counterpart at identical settings is
[`samples-tt-tnt-v3-t0.8.md`](samples-tt-tnt-v3-t0.8.md). Read them side by side.

**Verdict: not usable.** Sampling does remove the hard greedy repetition loops, but what it
reveals underneath is not coherent prose. Against the CPU output at the same settings the
device output is agrammatical (run-on clauses with no verb agreement, e.g. *"She lived in a
girl named Lily lived her family of the top number one day"*), emits malformed non-words that
never appear in the CPU output (*"Invisers"*, *"o'Splains"*, *"megathering"*, *"Micropurs"*,
*"d'Bule"*), and repeats locally at 1.55x the CPU rate (local-repeat rate 0.161 vs 0.104;
distinct-word ratio 0.633 vs 0.663 over the same 29 completions). The CPU output at these
settings is consistently grammatical and locally coherent even where it is semantically odd.

`⟨STOPPED⟩` marks a completion that ended on `</s>` rather than hitting the 60-token limit.

### voice-01 · _target-voice_

**sample 1/2:**

> The chimp chose the longest stick, then the one that had been** 5 in. The first started by the first; and the remaining the other. By-h and the second was 25163. The tail the more the longest did not above the lowest the size of the strongest to begin. (when the initial was made the right-forked out of the**

**sample 2/2:**

> The chimp chose the longest stick, then the one that had been** planted from the winner. He made. The plumba. This is the top number one foot, so much bigger. The red piece was the best to the second, while the rest five-ofte as high-he made his own piece The third. The length and the Four. The**

### voice-02 · _target-voice_

**sample 1/2:**

> The ants had learned that being eaten was a way of** cooking. The most important. The other ants had to be careful. They needed and the ants did not only the most of fuel they had to the antcorners did not been very well. The ants. Invisers. They were the ants have eaten not only used quickly. They were not**

**sample 2/2:** ⟨STOPPED⟩

> The ants had learned that being eaten was a way of** helping the way of the bestowing.**

### voice-03 · _target-voice_

**sample 1/2:**

> Once upon a time, there was a little** girl named Timmy. Timmy. Timmy. Timmy loved to go on a little girl named Timmy's birthday on a pony. Timmy loved to a big to a boy named Brownie. One little girl named Lily loved to a pony. One at the pony. One day, there was a pony named Lily's**

**sample 2/2:**

> Once upon a time, there was a little** girl named Lily. She lived in a girl named Lily lived her family of the top number one day, Lily was very well. She was a girl named Lily lived in the most brave and a beautiful and she loved to the world filled with her parents, a girl named Lily and she loved to a**

### stutter-01 · _stutter_

**sample 1/2:**

> A rose is a rose is a** flower in this is a real flower. The rose is the colour; the blossom and the flower and it is the flower that is pink and the flower. The flower is a rose is a flower is a rose is a very pale pink, it is purple. This is pink, a rose. The**

**sample 2/2:**

> A rose is a rose is a** rose, and a vowet, a good and a lady. A lyre, not a kiss, like the lady. It is a rose, She is not for the rose, where it is or a as my handhe a rose. A is a rose, a little. I love was**

### stutter-02 · _stutter_

**sample 1/2:**

> The little mouse. The little mouse. The** bird and the mouse had made the mouse was very happy. He loved the mouse and the mouse and the mouse felt a big, the mouse was sad. The mouse felt happy. The mouse. He could not sad to his mouse was not the mouse was not his mouse. The mouse. The mouse**

**sample 2/2:**

> The little mouse. The little mouse. The** mouse was very first is very happy. The mouse. The mouse had a dog was not knowing the mouse loved the mouse. The mouse had a mouse. The mouse. The mouse. They were very sad mouse. They all the mouse. The mouse. The mouse. The mouse. They all the**

### oracle-01 · _oracular_

**sample 1/2:**

> The question was whether to go. The answer came back:** "To what is, if you would you would be it is, I am quite well; I must look in a word or not. If the child must be a child I have I amer of this is to- Notice, it was, for not. I know it is—I**

**sample 2/2:**

> The question was whether to go. The answer came back:** "I am I've I'may in the lady has she had it'sn'a'n't the loss. It's never seen thee. We left me, she'st'the don't 'times, please goin'er didn't always a little God I've got**

### oracle-02 · _oracular_

**sample 1/2:**

> Above, the mountain. Below, the lake. The image is of** ancient and is not very realising; the shore. "thea:—To the gods of the planet, the lake which men's sun-to a boy-and Island.   “a; the image is to-of-times—the-is.   The Head of the father**

**sample 2/2:**

> Above, the mountain. Below, the lake. The image is of** the parenthesis. It is made of the plavea. It is the top of the gods are revealed towered. It is a group. The spell, for the gold symbols where the images and the city of the writing made of the Aztec kings are the city of the God. The wonder**

### agentic-01 · _agentic_

**sample 1/2:**

> I placed a straw across the trench and waited. The procession** was in the procession was started by was, they were broad enough; then came and the procession when the procession was a pair of the sun-to it must-and then out of the following them, at a tree-track at the burden of the whole procession. The procession of a procession.**

**sample 2/2:**

> I placed a straw across the trench and waited. The procession** of the next morning is at this window was followed the ground was covered it had brought in the village, and the next morning. The twelve hours were now to the village there was on the five miles away. In my two or even now going round the poor soil. The men were the very early**

### agentic-02 · _agentic_

**sample 1/2:**

> To find out what the bees would do, I first** saw a little dread it were in their honey; I put the little bees. There was to thinkin'a'ha or the sun-to it must-andit out o'Splains; In a-her-times, they were nothing, a bee-m**

**sample 2/2:**

> To find out what the bees would do, I first** to ask her question her at first. I know. She was the time she told me how I was, and why. She said, but I knew her master, and what I should I was or what she knew the bees and I spoke of my poor. She said and that I have done**

### ground-01 · _grounding_

**sample 1/2:**

> An ant is an insect that lives in** caves in dense forests. People who would not yet is the eagle, and needs the hawk is a moth, being a lot of various native to the need the bird to begets megathering, but is a very much like the first of its mate not only one of a much larger bird**

**sample 2/2:**

> An ant is an insect that lives in** swamps, and lives. Its insect. It lives in the main-related to fly its larvae. It can fly. The bug. The Micropurs can prey, there. It's eggs, or does not only in the insect. It can live in the insect that lives in the insect prey**

### ground-02 · _grounding_

**sample 1/2:**

> Chimpanzees use tools such as** d'Bule coins' blocks from a plover the eagle; and the giant crowns to make a flower, and hooks or 'a may also need the scales; but also cut down the size of this is a very strong (when it wasnt; not only the BD by the dried**

**sample 2/2:**

> Chimpanzees use tools such as** to use the skull. Thibet, make plated, and the bull's hammer, and which are used to make. It is a tool. Leves the most of the mainwanger, or creeps can easily, and other human tools used by the skull a little rod. The web**

### assoc-01 · _perpendicular_

**sample 1/2:** ⟨STOPPED⟩

> The stick remembered being a** kind and scary friend. The bear was not sharing is a good dog was a big. And when it had been a lot of a big.**

**sample 2/2:** ⟨STOPPED⟩

> The stick remembered being a** good friend and helped with a good at hand and tookr.**

### assoc-02 · _perpendicular_

**sample 1/2:**

> There is a kind of light that only falls** out of all at first. Somewhere, and is the sky, and needs and cannot be pictured in the night. A series. The long to a boylike to you can be seen. He is very difficult to-up (when it was the light not only the skyless a little do**

**sample 2/2:**

> There is a kind of light that only falls**. The rainbow-colored at this color in the clouds that is the night. The number one day and day, with a little red, but the color to the light there is the rest. The object. The light. One night. The light is a dark night light and the night and light**

### long-01 · _coherence_

**sample 1/2:**

> The old woman kept bees behind the house, and every morning she** came in her grandmother had to watch was gone; and the old nurse whom she found the little ones were all night; and kept the sun-born child at nightnit did not only the day-time; and night-and, and the day; but still she had a voice of the**

**sample 2/2:**

> The old woman kept bees behind the house, and every morning she** would take a hive was alone, so busy and took up and the bees had the bees and the bees, and learned. The bees had seen the bee, and for the hive bees that honey-making. She knew the bees filled all the hive filled with the bees spent the bees, and was**

### long-02 · _coherence_

**sample 1/2:**

> In the winter the pond froze, and the children**'s fish were very hungry. The fish were not far away from the pond. But the pond and the children in a lot was frozen and the baby fish and the ducks and the little duck. The children were eaten to pieces. They all the children was eaten not the sun-Ding were dried**

**sample 2/2:** ⟨STOPPED⟩

> In the winter the pond froze, and the children** ran up and the lake began to their mother and the ducks.**
