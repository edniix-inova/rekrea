Create a new Remotion composition for a 5-second vertical social clip.
Use the Remotion best-practices skill.

## Specs
- Composition id: TorqueIntro
- 1080×1920, 30 fps, 150 frames
- All animation must be driven by useCurrentFrame() with interpolate() or spring().
  No CSS transitions or CSS keyframe animations.
- Put every spring config and timing value as named constants at the top of the
  file so I can tune them in one place.
- Expose the title and label text as props with a zod schema so I can edit them
  in Remotion Studio.

## Style
- Background #0E0F12, text #F2F1EC, one accent color #FF6A2B. No other colors.
- Font: Inter via @remotion/google-fonts. Title weight 800, label weight 400.
- Keep all text and graphics inside the central area: nothing in the top 250 px
  or bottom 400 px (platform UI overlays sit there).

## Beat timeline (frames)
1. 0–25: Title "TORQUE" [SWAP] enters letter by letter from 60 px below with
   opacity 0→1. Spring: stiffness 120, damping 14, mass 1. Stagger 3 frames per letter.
2. 25–75: Below the title, a simple smooth curve (rising hump, like a torque
   curve [SWAP]) draws itself left to right using SVG stroke-dashoffset.
   Ease-in-out (Easing.inOut(Easing.cubic)) over 50 frames. Accent color, 6 px stroke.
3. 75–85: A dot pops in at the curve's peak. Spring: stiffness 200, damping 8
   (deliberately bouncy).
4. 80–95: Label "peak" [SWAP] slides up 20 px and fades in next to the dot.
   Ease-out (Easing.out(Easing.cubic)) over 15 frames.
5. 95–130: Hold. The dot pulses gently: scale 1→1.08→1, one sine cycle.
6. 130–150: Exit. All elements fade out and scale to 0.95 with ease-in
   (Easing.in(Easing.cubic)) over 12 frames, staggered 3 frames in the order:
   label, dot, curve, title. Frame 149 must be an empty background so the
   clip loops seamlessly.

## Verification
- Start Remotion Studio so I can scrub the timeline.
- Render stills at frames 0, 20, 75, 120 and 149 and check them yourself
  against the timeline above. Fix mismatches before reporting back.
- Render the final MP4 to out/torque-intro.mp4.

## Learning
When done, explain in max 10 lines which Remotion primitives you used
(Sequence, interpolate, spring, Easing, etc.) and where in the file each one
lives, so I can tune it myself.