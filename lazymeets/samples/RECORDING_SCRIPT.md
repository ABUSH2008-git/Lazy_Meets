# Script for recording a sample meeting

This is for the "meeting recording that may be shared" deliverable. Grab two friends and record this on a phone in a quiet room. It takes about 3 minutes. Use your own names if you like, but then use the same names in the app's *Participants* box.

You don't have to read it word for word. Small slips and "um"s make it more realistic. Just keep the numbers, the "not"s, and who says "I'll do it" the same, because those are what the app gets checked on.

**People:** Rahul (leads the meeting), Sneha, Karthik

---

**Rahul:** Okay, let's start. This is our ML team sync for the bootcamp project. Sneha, Karthik, thanks for joining. Three things today: the training run, the evaluation numbers, and the submission on the seventeenth.

**Sneha:** Sure. So the LoRA fine-tune on the Llama checkpoint finished last night. It took about six hours on the A100, and we hit CUDA out of memory twice before I dropped the batch size to sixteen.

**Karthik:** I saw the Weights and Biases dashboard. The validation loss is still going down, so we're not overfitting yet.

**Rahul:** Nice. What are the numbers?

**Sneha:** F1 score is zero point eight one on the validation set. The baseline was zero point seven two. Precision is fine, recall is the weak part.

**Karthik:** I'd like to try quantizing it to four bits with ONNX so it runs on CPU for the demo.

**Rahul:** Good idea. Let's go with ONNX export for the demo build. That's decided.

**Sneha:** There was also the idea of switching the whole pipeline to JAX.

**Karthik:** Honestly, I don't think that's worth it. We don't have time to rewrite everything.

**Rahul:** Agreed, we're not moving to JAX. We stay on PyTorch.

**Rahul:** Okay, action items. Karthik, can you do the ONNX export and benchmark it by Friday?

**Karthik:** Yes, I'll have the ONNX export and the latency numbers by Friday.

**Sneha:** I'll write the evaluation section of the report.

**Rahul:** Thanks. We also need to clean up the GitHub README before the submission.

**Karthik:** Yeah, someone has to do that.

**Rahul:** And maybe Ananya could help with the demo video, but I'll have to ask her first.

**Sneha:** One open question. Do we have enough Colab credits left for another full training run? It costs around nine hundred rupees.

**Rahul:** I don't know yet. I'll check the credits and tell you by tomorrow evening.

**Karthik:** One last thing. The tokenizer warning in the logs is not a real bug, so let's ignore it for now.

**Rahul:** Fine, we'll ignore it. Submission is still on the seventeenth. Thanks, everyone.

---

## What the app should produce

Use this to check the output. The real output will be worded differently.

**Decisions**
1. Use ONNX export (4-bit quantized, CPU) for the demo build
2. Don't move to JAX; stay on PyTorch
3. Ignore the tokenizer warning for now (not a real bug)

**Proposals not agreed**
- Switch the pipeline to JAX (rejected)

**Action items**
| Task | Owner | Deadline |
|---|---|---|
| ONNX export + latency benchmark | Karthik | by Friday |
| Write the evaluation section of the report | Sneha *(only with speaker detection on; otherwise Unspecified)* | Unspecified |
| Clean up the GitHub README | Unspecified | before the submission |
| Check remaining Colab credits | Rahul *(only with speaker detection on; nobody says "Rahul" there)* | by tomorrow evening |

**Suggested follow-ups (not confirmed)**
- Help with the demo video (Ananya was only suggested; Rahul has to ask her first)

**Open questions**
- Are there enough Colab credits for another full training run (about ₹900)?

**Things the transcript must keep right**
- Numbers: six hours, batch size 16, F1 0.81, baseline 0.72, 4 bits, ₹900, the 17th
- Negations: "not overfitting", "don't think that's worth it", "don't have time", "not moving to JAX", "don't know yet", "not a real bug"
- Terms the refinement should get right: LoRA, Llama, A100, CUDA, Weights and Biases, F1, ONNX, JAX, PyTorch, Colab, GitHub README

## Settings to use in the app

- **Participants:** Rahul, Sneha, Karthik, Ananya
- **Glossary:** LoRA, Llama, A100, CUDA, Weights and Biases, ONNX, JAX, PyTorch, Colab
- **Domain:** Machine learning / AI
- Turn on **Detect who is speaking** and set speakers to 3. Then name the speakers in the Speakers tab and click *Save names & regenerate minutes*.

## Recording tips

- One phone in the middle of the table works fine. Keep fans and AC noise down.
- Most phone recorders save .m4a. That's fine, upload it as is.
- Talk at a normal pace and don't talk over each other too much.
- Save the file as `samples/team_sync.m4a`.

## Making the sample outputs for the submission

Easiest is the web app, since it's the same thing you show in the demo video. Process the recording with the settings above, name the speakers, regenerate the minutes, then click **Download all**. Unzip it into `samples/outputs/`.

Or use the command line (it can't name speakers, so self-assigned tasks stay Unspecified):

```bash
python cli.py samples/team_sync.m4a --participants "Rahul, Sneha, Karthik, Ananya" \
  --glossary "LoRA, Llama, A100, CUDA, Weights and Biases, ONNX, JAX, PyTorch, Colab" \
  --domain "Machine learning / AI" --diarize --speakers 3 --out samples/outputs
```
