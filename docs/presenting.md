# Presenting ClearSAR

The deck is `docs/ClearSAR_Final_Presentation.pptx`: 20 main slides plus 3 backup slides. Every slide has speaker notes, so use Presenter View. It is rebuilt from `docs/deck_src/` (run `make_assets.py`, then `make_bg.py`, then `node build_deck.js`).

## Timing (about 15 minutes, then questions)

| Slides | Part | Minutes |
|---|---|---|
| 1-3 | Problem and task | 2 |
| 4-6 | Data, Phase 1 lessons, the date shift | 3 |
| 7-12 | Model, gate, training, GAN, engineering | 4.5 |
| 13-16 | Results | 3 |
| 17-18 | Expert system and **live demo** | 2 (demo 3 if time allows) |
| 19-20 | Limits and takeaways | 0.5 |

## Before the talk

1. Start the app (`.venv\Scripts\python.exe -m streamlit run app.py`) and open the **Advisor** page once, so its analyses are computed and cached. The first run takes about a minute on a CPU.
2. Keep the browser on the Home page in a second window. Slide 18 has a screenshot to fall back on if the demo fails.

## Demo path (3 minutes)

1. **Home**: drag the slider across the Indian test patch.
2. **Results**: switch to "by cloud cover" and point at the 90-100 % bin.
3. **Explore patches**: pick the Portugal patch (81 % cloud) and show its gate map.
4. **Advisor**: same patch. Change the intended use from *visual inspection* to *vegetation monitoring*; the recommended model changes from the GAN model to the before-GAN model, and the rule that fired is shown.
5. **Knowledge base**: in the what-if simulator, move the cloud slider and watch the trust level change.

## Story to tell

- The thread: Phase 1 produced lessons, not a winner. Each lesson became a design decision (slide 5). Keep coming back to "measured, not guessed".
- The strongest single moment is slide 8: nobody told the gate where the clouds are.
- Say the limitations before you are asked (slides 16 and 19). Examiners reward that.
- Quote the GAN result honestly: sharper, with no *measurable* PSNR cost. By eye the change is subtle.
- The backup slides hold the hyper-parameters, the Phase 1 numbers and short answers to likely questions.
