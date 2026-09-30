import {Composition} from 'remotion';
import {HelloTitle, helloTitleSchema} from './compositions/HelloTitle/HelloTitle';
import {
  TorqueIntro,
  torqueIntroSchema,
  TORQUE_INTRO_DURATION,
  TORQUE_INTRO_HEIGHT,
  TORQUE_INTRO_WIDTH,
} from './compositions/TorqueIntro/TorqueIntro';

export const FPS = 30;

// Register every composition here. One folder per motion graphic under src/compositions/.
export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="HelloTitle"
        component={HelloTitle}
        schema={helloTitleSchema}
        durationInFrames={3 * FPS}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{title: 'Rekrea', subtitle: 'Motion graphics with Remotion'}}
      />
      <Composition
        id="TorqueIntro"
        component={TorqueIntro}
        schema={torqueIntroSchema}
        durationInFrames={TORQUE_INTRO_DURATION}
        fps={FPS}
        width={TORQUE_INTRO_WIDTH}
        height={TORQUE_INTRO_HEIGHT}
        defaultProps={{title: 'TORQUE', label: 'peak'}}
      />
    </>
  );
};
