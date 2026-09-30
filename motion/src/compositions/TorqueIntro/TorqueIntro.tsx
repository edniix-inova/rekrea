import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Inter';
import {z} from 'zod';

const {fontFamily} = loadFont('normal', {weights: ['400', '800'], subsets: ['latin']});

// ---------------------------------------------------------------------------
// Tunable constants (all times are frames at 30 fps)
// ---------------------------------------------------------------------------

// Composition
export const TORQUE_INTRO_WIDTH = 1080;
export const TORQUE_INTRO_HEIGHT = 1920;
export const TORQUE_INTRO_DURATION = 150;

// Palette
const COLOR_BG = '#0E0F12';
const COLOR_TEXT = '#F2F1EC';
const COLOR_ACCENT = '#FF6A2B';

// Safe area: platform UI sits above/below these lines
const SAFE_TOP = 250;
const SAFE_BOTTOM = 400;
const SAFE_CENTER_Y = (SAFE_TOP + (TORQUE_INTRO_HEIGHT - SAFE_BOTTOM)) / 2;

// Layout
const TITLE_FONT_SIZE = 190;
const TITLE_WEIGHT = 800;
const TITLE_HEIGHT = 230;
const LABEL_FONT_SIZE = 56;
const LABEL_WEIGHT = 400;
const CURVE_WIDTH = 880;
const CURVE_HEIGHT = 400;
const CURVE_STROKE = 6;
const GAP_TITLE_CURVE = 60;
const DOT_RADIUS = 22;
const LABEL_OFFSET_X = 36; // label left edge, right of the dot centre
const LABEL_OFFSET_Y = 62; // label centre, above the dot (keeps it clear of the curve)

// Beat 1: title, letter by letter
const TITLE_STAGGER = 3;
const TITLE_RISE_PX = 60;
const TITLE_SPRING = {stiffness: 120, damping: 14, mass: 1};

// Beat 2: curve draws itself
const CURVE_START = 25;
const CURVE_DURATION = 50;
const CURVE_EASING = Easing.inOut(Easing.cubic);

// Beat 3: dot pops in at the peak
const DOT_START = 75;
const DOT_SPRING = {stiffness: 200, damping: 8, mass: 1};

// Beat 4: label slides up and fades in
const LABEL_START = 80;
const LABEL_DURATION = 15;
const LABEL_RISE_PX = 20;
const LABEL_EASING = Easing.out(Easing.cubic);

// Beat 5: dot pulse, one full cycle 1 -> 1.08 -> 1
const PULSE_START = 95;
const PULSE_END = 130;
const PULSE_SCALE = 1.08;

// Beat 6: exit. Order: label, dot, curve, title.
// Frame 149 must be empty: the last element finishes at
// EXIT_START + 3 * EXIT_STAGGER + EXIT_DURATION = 149.
const EXIT_START = 128;
const EXIT_DURATION = 12;
const EXIT_STAGGER = 3;
const EXIT_SCALE = 0.95;
const EXIT_EASING = Easing.in(Easing.cubic);
const EXIT_ORDER = {label: 0, dot: 1, curve: 2, title: 3};

// ---------------------------------------------------------------------------
// Curve geometry (local coordinates inside the CURVE_WIDTH x CURVE_HEIGHT box)
// ---------------------------------------------------------------------------
const PEAK_X = CURVE_WIDTH * 0.55;
const PEAK_Y = CURVE_HEIGHT * 0.15;
const BASE_Y = CURVE_HEIGHT * 0.85;
// Horizontal tangents at the peak, so the peak is exactly (PEAK_X, PEAK_Y).
const CURVE_PATH = [
  `M 0 ${BASE_Y}`,
  `C ${PEAK_X * 0.4} ${BASE_Y} ${PEAK_X * 0.55} ${PEAK_Y} ${PEAK_X} ${PEAK_Y}`,
  `C ${PEAK_X * 1.45} ${PEAK_Y} ${CURVE_WIDTH * 0.85} ${BASE_Y - 40} ${CURVE_WIDTH} ${BASE_Y}`,
].join(' ');

// ---------------------------------------------------------------------------
// Props (editable in Remotion Studio)
// ---------------------------------------------------------------------------
export const torqueIntroSchema = z.object({
  title: z.string(),
  label: z.string(),
});
export type TorqueIntroProps = z.infer<typeof torqueIntroSchema>;

/** Exit progress 0 -> 1 for one element, eased. */
const useExit = (frame: number, order: number) => {
  const start = EXIT_START + order * EXIT_STAGGER;
  const t = interpolate(frame, [start, start + EXIT_DURATION], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: EXIT_EASING,
  });
  return {opacity: 1 - t, scale: 1 - t * (1 - EXIT_SCALE)};
};

export const TorqueIntro: React.FC<TorqueIntroProps> = ({title, label}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  // Vertical layout: title + gap + curve, centred in the safe area
  const blockHeight = TITLE_HEIGHT + GAP_TITLE_CURVE + CURVE_HEIGHT;
  const titleTop = SAFE_CENTER_Y - blockHeight / 2;
  const curveTop = titleTop + TITLE_HEIGHT + GAP_TITLE_CURVE;
  const curveLeft = (TORQUE_INTRO_WIDTH - CURVE_WIDTH) / 2;

  const titleExit = useExit(frame, EXIT_ORDER.title);
  const curveExit = useExit(frame, EXIT_ORDER.curve);
  const dotExit = useExit(frame, EXIT_ORDER.dot);
  const labelExit = useExit(frame, EXIT_ORDER.label);

  // Beat 2: curve draw progress (pathLength=1 lets us use 0..1 for the dash)
  const curveProgress = interpolate(frame, [CURVE_START, CURVE_START + CURVE_DURATION], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: CURVE_EASING,
  });

  // Beat 3: dot pop
  const dotPop = spring({frame: frame - DOT_START, fps, config: DOT_SPRING});

  // Beat 5: pulse, one smooth cycle 1 -> PULSE_SCALE -> 1
  const pulseT = interpolate(frame, [PULSE_START, PULSE_END], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const pulse = 1 + (PULSE_SCALE - 1) * ((1 - Math.cos(2 * Math.PI * pulseT)) / 2);
  const dotRadius = DOT_RADIUS * dotPop * pulse * dotExit.scale;

  // Beat 4: label
  const labelT = interpolate(frame, [LABEL_START, LABEL_START + LABEL_DURATION], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: LABEL_EASING,
  });

  const curveCx = CURVE_WIDTH / 2;
  const curveCy = CURVE_HEIGHT / 2;

  return (
    <AbsoluteFill style={{backgroundColor: COLOR_BG, fontFamily, color: COLOR_TEXT}}>
      {/* Beat 1: title, letter by letter */}
      <div
        style={{
          position: 'absolute',
          top: titleTop,
          left: 0,
          width: TORQUE_INTRO_WIDTH,
          height: TITLE_HEIGHT,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          opacity: titleExit.opacity,
          transform: `scale(${titleExit.scale})`,
        }}
      >
        {title.split('').map((char, i) => {
          const s = spring({frame: frame - i * TITLE_STAGGER, fps, config: TITLE_SPRING});
          return (
            <span
              key={i}
              style={{
                display: 'inline-block',
                fontSize: TITLE_FONT_SIZE,
                fontWeight: TITLE_WEIGHT,
                lineHeight: 1,
                whiteSpace: 'pre',
                opacity: Math.min(Math.max(s, 0), 1),
                transform: `translateY(${(1 - s) * TITLE_RISE_PX}px)`,
              }}
            >
              {char}
            </span>
          );
        })}
      </div>

      {/* Beats 2, 3, 5: curve and dot */}
      <svg
        width={CURVE_WIDTH}
        height={CURVE_HEIGHT}
        viewBox={`0 0 ${CURVE_WIDTH} ${CURVE_HEIGHT}`}
        style={{position: 'absolute', top: curveTop, left: curveLeft, overflow: 'visible'}}
      >
        {frame >= CURVE_START && (
          <g
            opacity={curveExit.opacity}
            transform={`translate(${curveCx} ${curveCy}) scale(${curveExit.scale}) translate(${-curveCx} ${-curveCy})`}
          >
            <path
              d={CURVE_PATH}
              pathLength={1}
              fill="none"
              stroke={COLOR_ACCENT}
              strokeWidth={CURVE_STROKE}
              strokeLinecap="round"
              strokeDasharray={1}
              strokeDashoffset={1 - curveProgress}
            />
          </g>
        )}
        {frame >= DOT_START && (
          <circle cx={PEAK_X} cy={PEAK_Y} r={Math.max(dotRadius, 0)} fill={COLOR_ACCENT} opacity={dotExit.opacity} />
        )}
      </svg>

      {/* Beat 4: label next to the dot */}
      <div
        style={{
          position: 'absolute',
          left: curveLeft + PEAK_X + LABEL_OFFSET_X,
          top: curveTop + PEAK_Y - LABEL_OFFSET_Y - LABEL_FONT_SIZE / 2,
          height: LABEL_FONT_SIZE,
          fontSize: LABEL_FONT_SIZE,
          fontWeight: LABEL_WEIGHT,
          lineHeight: 1,
          whiteSpace: 'pre',
          opacity: labelT * labelExit.opacity,
          transform: `translateY(${(1 - labelT) * LABEL_RISE_PX}px) scale(${labelExit.scale})`,
          transformOrigin: 'left center',
        }}
      >
        {label}
      </div>
    </AbsoluteFill>
  );
};
