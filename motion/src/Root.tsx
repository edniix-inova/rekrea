import {Composition} from 'remotion';
import {HelloTitle, helloTitleSchema} from './compositions/HelloTitle/HelloTitle';

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
    </>
  );
};
