# Rekrea Motion

Motion graphics for Rekrea, built with [Remotion](https://www.remotion.dev/) (React). This is a self-contained Node project, independent from the Python package in `rekrea/`.

## Setup

Requires Node.js 18+.

```bash
cd motion
npm install
```

## Usage

```bash
npm run studio                                    # interactive preview
npm run render -- HelloTitle out/hello.mp4        # render a composition
npm run render -- HelloTitle out/hello.mp4 --props='{"title":"Hi","subtitle":"There"}'
npm run still -- HelloTitle out/hello.png         # single frame
npm run typecheck
```

### Transparent output (to overlay on footage)

```bash
npm run render -- HelloTitle out/hello.mov --codec=prores --prores-profile=4444 --pixel-format=yuva444p10le
```

## Structure

```
motion/
├── src/
│   ├── index.ts                 # registerRoot
│   ├── Root.tsx                 # register compositions here
│   ├── compositions/<Name>/     # one folder per motion graphic
│   └── components/              # shared building blocks
├── public/                      # assets loaded via staticFile()
└── out/                         # renders (git-ignored)
```

## Adding a composition

1. Create `src/compositions/<Name>/<Name>.tsx`.
2. Register it in `src/Root.tsx` with a `<Composition id="<Name>" ... />`.
3. Preview with `npm run studio`.

## Using Rekrea outputs

Copy or symlink outputs from the Rekrea base environment (e.g. transparent frames from background removal, or enhanced clips) into `public/` and load them with `staticFile()`, `<Img>` or `<OffthreadVideo>`. Renders can be written back to the base environment's `motion_graphics/remotion/output` folder.
