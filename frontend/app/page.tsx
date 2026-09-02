import { Console } from "./components/Console";
import { FieldCanvas } from "./components/FieldCanvas";
import { Go, TopBar } from "./components/Nav";
import { Plays } from "./components/Plays";
import { Scoreboard } from "./components/Scoreboard";
import { Splash } from "./components/Splash";

export default function Page() {
  return (
    <>
      <Splash />
      <TopBar />

      <main id="view-home">
        <div className="hero">
          <FieldCanvas />
          <div className="wrap">
            <p className="kicker">
              Start / sit, settled · <b>2025 season</b>
            </p>
            <h1>
              Who do <span className="out">I</span> start?
            </h1>
            <p className="sub">
              Two players. One lineup spot. Kickoff in an hour and everyone on the
              internet has a different answer. <b>Pick both. Get one.</b>
            </p>
            <div className="hero-cta">
              <Go to="app">Set your lineup</Go>
              <Go to="how" className="btn ghost">
                How it calls it
              </Go>
            </div>
          </div>
        </div>

        <Scoreboard />

        <section id="how">
          <div className="wrap">
            <p className="eyebrow">The play</p>
            <h2>Three steps, in that order</h2>
            <p className="lead">
              The order matters more than it sounds. Each step is walled off from the
              one before it, so nothing downstream can quietly talk the number into
              something friendlier.
            </p>
            <Plays />
          </div>
        </section>

        <div className="band">
          <section>
            <div className="wrap">
              <blockquote>
                The model does the math. The writer just <em>explains</em> it.
              </blockquote>
              <p>
                Anything that can write you a confident paragraph can write you a
                confident number too. Here they are different things, and only one of
                them is allowed near your lineup.
              </p>
            </div>
          </section>
        </div>

        <section>
          <div className="wrap">
            <p className="eyebrow">No takebacks</p>
            <h2>Every call gets graded</h2>
            <p className="lead">
              These are completed games, so the truth is already on record. Every
              recommendation shows you what actually happened afterwards, including
              the ones it got wrong. A projection you cannot check is just a vibe with
              a decimal point.
            </p>
            <div className="hero-cta">
              <Go to="app">Pick a matchup</Go>
            </div>
          </div>
        </section>
      </main>

      <Console />

    </>
  );
}
