//! Adapted from cold-clear/tbp/src/bin/cc_replay.rs (Cold Clear, MPL-2.0).
//! Plays Cold Clear forward from recorded positions under many piece sequences, and prints what it
//! sent on each piece, for nana's scripts/opponent/cc_predictability.py.
//! usage: cc_replay --positions FILE [--horizon H] [--thinks N] [--queue-seeds Q] [--repeats R]
//!                  [--seed B] [--jobs J]
//! FILE: {"positions": [{board, queue, hold, bag, b2b, combo, meter, incoming, future?}, ...]}: board is 40 rows,
//! bottom first, of 0/1; queue is [current, preview...]; bag is what is left of the 7-bag after the
//! queue; future (optional) is the pieces the game really dealt next. Each position plays R
//! rollouts for each of Q random continuations of the bag, and R more on `future` (queue seed -1)
//! when it has one. Rollouts on the same queue differ only by the search's own randomness.
//! Fixed incoming attacks arrive unless --no-garbage 1 ignores both incoming and initial meter.
//! One JSON line per rollout on stdout.
//! Build with cargo build --release --manifest-path scripts/opponent/opponent_replay/Cargo.toml.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};

use cold_clear::evaluation::Standard;
use cold_clear::{BotState, Options};
use libtetris::*;
use serde_json::{json, Value};

fn piece(name: &str) -> Piece {
    match name {
        "I" => Piece::I,
        "O" => Piece::O,
        "T" => Piece::T,
        "L" => Piece::L,
        "J" => Piece::J,
        "S" => Piece::S,
        "Z" => Piece::Z,
        _ => panic!("unknown piece {}", name),
    }
}

fn piece_name(p: Piece) -> &'static str {
    match p {
        Piece::I => "I",
        Piece::O => "O",
        Piece::T => "T",
        Piece::L => "L",
        Piece::J => "J",
        Piece::S => "S",
        Piece::Z => "Z",
    }
}

fn pieces(v: &Value) -> Vec<Piece> {
    v.as_array()
        .map(|a| a.iter().map(|p| piece(p.as_str().unwrap())).collect())
        .unwrap_or_default()
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

struct Position {
    board: Board,
    future: Option<Vec<Piece>>,
    incoming: Vec<u32>,
    meter: u32,
}

fn load(j: &Value) -> Position {
    let mut board = Board::new();
    let mut field = [[false; 10]; 40];
    for (y, row) in j["board"].as_array().unwrap().iter().enumerate() {
        for (x, cell) in row.as_array().unwrap().iter().enumerate() {
            field[y][x] = cell.as_u64().unwrap() != 0;
        }
    }
    board.set_field(field);
    board.hold_piece = j["hold"].as_str().map(piece);
    for p in pieces(&j["queue"]) {
        board.add_next_piece(p);
    }
    let bag = pieces(&j["bag"]);
    if !bag.is_empty() {
        board.bag = bag.into_iter().collect();
    }
    board.combo = j["combo"].as_u64().unwrap() as u32;
    board.b2b_bonus = j["b2b"].as_u64().unwrap() as u32;
    let future = j.get("future").filter(|f| f.is_array()).map(pieces);
    let incoming = j["incoming"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    let meter = j["meter"].as_u64().unwrap() as u32;
    Position {
        board,
        future,
        incoming,
        meter,
    }
}

/// lock a placement and take its piece off the queue, through hold when it isn't the current piece
fn advance(board: &mut Board, placement: FallingPiece) -> LockResult {
    let result = board.lock_piece(placement);
    let next = board.advance_queue().unwrap();
    if next != placement.kind.0 {
        let unheld = board.hold(next);
        let p = unheld.unwrap_or_else(|| board.advance_queue().unwrap());
        assert_eq!(p, placement.kind.0);
    }
    result
}

fn pressure(meter: u32, attack: u32, incoming: u32, cleared: bool) -> (u32, u32) {
    let cancelled = meter.min(attack);
    let meter = meter - cancelled + incoming.saturating_sub(attack - cancelled);
    if cleared {
        (meter, 0)
    } else {
        (0, meter)
    }
}

fn rollout(
    pos: &Position,
    position_id: usize,
    queue_seed: i64,
    repeat: usize,
    base_seed: u64,
    horizon: usize,
    thinks: u32,
    no_garbage: bool,
) -> Vec<Value> {
    let eval = Standard::default();
    let mut game = pos.board.clone();
    let queue_len = game.next_queue().count();
    let options = Options {
        speculate: true,
        threads: 1,
        min_thinks: if no_garbage { Options::default().min_thinks } else { thinks },
        max_thinks: thinks,
        ..Default::default()
    };
    let mut bot = BotState::new(pos.board.clone(), options);
    let mut rng = base_seed
        ^ (queue_seed as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ if no_garbage { 0 } else { (position_id as u64).wrapping_mul(0x94D049BB133111EB) };
    // Separate streams: repeats share the queue, but resample unknown garbage holes.
    let mut garbage_rng = rng ^ (repeat as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let mut meter = if no_garbage { 0 } else { pos.meter };
    let mut dealt = 0;
    let mut moves = vec![];
    let steps = if no_garbage { horizon } else { horizon.min(pos.incoming.len()) };
    for step in 0..steps {
        let mut stuck = 0;
        loop {
            match bot.think() {
                Ok(thinker) => {
                    let result = thinker.think(&eval);
                    bot.finish_thinking(result);
                }
                Err(true) => {
                    stuck += 1;
                    if stuck > 16 {
                        break;
                    }
                }
                Err(false) => break,
            }
        }
        let loc = match bot.suggest_move(&eval, None, 0) {
            Some((mv, _)) => mv.expected_location,
            None => break,
        };
        let current = game.next_queue().next().unwrap();
        let lock = advance(&mut game, loc);
        moves.push(json!({
            "piece": piece_name(loc.kind.0),
            "hold": loc.kind.0 != current,
            "attack": lock.garbage_sent,
            "lines": lock.cleared_lines.len(),
            "kind": format!("{:?}", lock.placement_kind),
            "pc": lock.perfect_clear,
            "b2b": game.b2b_bonus,
            "combo": game.combo,
            "height": game.column_heights().iter().max().copied().unwrap_or(0),
        }));
        if lock.locked_out {
            break;
        }
        bot.advance_move(loc);
        // Zero-delay stadium rules: own attack cancels the old meter first, then the
        // residual outgoing attack meets this turn's fixed external attack head-on.
        let (next_meter, tanked) = pressure(
            meter,
            lock.garbage_sent,
            if no_garbage { 0 } else { pos.incoming[step] },
            !lock.cleared_lines.is_empty(),
        );
        meter = next_meter;
        let hole = (splitmix(&mut garbage_rng) % 10) as usize;
        let mut dead = false;
        for _ in 0..tanked {
            dead |= game.add_garbage(hole);
        }
        if dead {
            break;
        }
        while game.next_queue().count() < queue_len {
            let p = if queue_seed < 0 {
                match pos.future.as_ref().and_then(|f| f.get(dealt)) {
                    Some(&p) => p,
                    None => break,
                }
            } else {
                let choices: Vec<Piece> = game.bag.iter().collect();
                choices[(splitmix(&mut rng) % choices.len() as u64) as usize]
            };
            dealt += 1;
            game.add_next_piece(p);
            bot.add_next_piece(p);
        }
        if tanked > 0 {
            // Stadium restarts the bot after garbage; retain the physical game state.
            bot = BotState::new(game.clone(), options);
        }
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_garbage_ignores_initial_meter_and_recorded_schedule_length() {
        let pos = load(&json!({
            "board": vec![vec![0u32; 10]; 40],
            "queue": ["I", "O", "T", "L", "J", "S"],
            "hold": null, "bag": ["Z"], "combo": 0, "b2b": 0,
            "meter": 100, "incoming": []
        }));
        assert!(rollout(&pos, 0, 0, 0, 1, 3, 1, false).is_empty());
        let moves = rollout(&pos, 0, 0, 0, 1, 3, 1, true);
        assert_eq!(moves.len(), 3);
        assert!(moves.iter().all(|m| m["height"].as_u64().unwrap() <= 12));
    }

    #[test]
    fn fixed_pressure_cancels_old_meter_before_simultaneous_attack() {
        assert_eq!(pressure(5, 3, 4, true), (6, 0));
        assert_eq!(pressure(2, 7, 4, true), (0, 0));
        assert_eq!(pressure(2, 7, 8, true), (3, 0));
    }

    #[test]
    fn zero_clear_tanks_entire_meter() {
        assert_eq!(pressure(10, 0, 4, false), (0, 14));
        assert_eq!(pressure(10, 0, 4, true), (14, 0));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut file = None;
    let mut horizon = 8usize;
    let mut thinks = 4000u32;
    let mut queue_seeds = 16i64;
    let mut repeats = 4usize;
    let mut base_seed = 1u64;
    let mut jobs = 1usize;
    let mut no_garbage = false;
    let mut i = 1;
    while i < args.len() {
        if i + 1 >= args.len() {
            panic!("{} needs a value", args[i]);
        }
        let v = &args[i + 1];
        match args[i].as_str() {
            "--positions" => file = Some(v.clone()),
            "--horizon" => horizon = v.parse().unwrap(),
            "--thinks" => thinks = v.parse().unwrap(),
            "--queue-seeds" => queue_seeds = v.parse().unwrap(),
            "--repeats" => repeats = v.parse().unwrap(),
            "--seed" => base_seed = v.parse().unwrap(),
            "--jobs" => jobs = v.parse::<usize>().unwrap().max(1),
            "--no-garbage" => no_garbage = v.parse::<u32>().unwrap() != 0,
            other => panic!("unknown option {}", other),
        }
        i += 2;
    }
    let text = std::fs::read_to_string(file.expect("--positions FILE")).unwrap();
    let j: Value = serde_json::from_str(text.trim_start_matches('\u{feff}')).unwrap();
    let positions: Arc<Vec<Position>> = Arc::new(
        j["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(load)
            .collect(),
    );

    let mut tasks = vec![];
    for (p, pos) in positions.iter().enumerate() {
        let first = if pos.future.is_some() { -1 } else { 0 };
        for q in first..queue_seeds {
            for r in 0..repeats {
                tasks.push((p, q, r));
            }
        }
    }
    let tasks = Arc::new(tasks);
    let next = Arc::new(AtomicUsize::new(0));
    let (send, receive) = mpsc::channel();
    let workers: Vec<_> = (0..jobs)
        .map(|_| {
            let (tasks, next, positions, send) =
                (tasks.clone(), next.clone(), positions.clone(), send.clone());
            std::thread::spawn(move || loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let (p, q, r) = match tasks.get(k) {
                    Some(&t) => t,
                    None => break,
                };
                let moves = rollout(&positions[p], p, q, r, base_seed, horizon, thinks, no_garbage);
                let line = json!({ "position": p, "queue_seed": q, "repeat": r, "moves": moves });
                send.send(line.to_string()).unwrap();
            })
        })
        .collect();
    drop(send);
    for line in receive {
        println!("{}", line);
    }
    for w in workers {
        w.join().unwrap();
    }
}
