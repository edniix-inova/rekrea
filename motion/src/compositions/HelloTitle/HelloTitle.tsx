import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';

// Props can be passed at render time: --props='{"title":"..."}' (schema is plain TS type here;
// swap for zod later if you want editable props in the Studio sidebar).
export type HelloTitleProps = {title: string; subtitle: string};
export const helloTitleSchema = undefined;

export const HelloTitle: React.FC<HelloTitleProps> = ({title, subtitle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const scale = spring({frame, fps, config: {damping: 12}});
  const subtitleOpacity = interpolate(frame, [20, 40], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  return (
    <AbsoluteFill
      style={{
        // Transparent background: render with --codec=prores --prores-profile=4444 or png sequence to overlay on footage.
        justifyContent: 'center',
        alignItems: 'center',
        fontFamily: 'sans-serif',
        color: 'white',
      }}
    >
      <h1 style={{fontSize: 160, margin: 0, transform: `scale(${scale})`}}>{title}</h1>
      <p style={{fontSize: 48, margin: 0, opacity: subtitleOpacity}}>{subtitle}</p>
    </AbsoluteFill>
  );
};
