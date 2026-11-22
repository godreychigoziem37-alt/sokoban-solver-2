// Sokoban solver for sokoban-solver-01: exact push-optimal solutions, room by room.
// The generator's levels are a hallway (the player's row) with rooms off it; a box never leaves its room, so each room
// is an independent problem. Each room is solved by a layered breadth-first search over pushes (walking is free), with
// dead-square pruning, so the pushes found are the room's optimum and the level's total is the sum of the rooms'.
// Deterministic: no clock, no randomness, no environment beyond the input and output directories.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_STATES = 1_000_000;

// The grid, padded by one wall on every side so neighbour arithmetic never wraps.
function parse(text) {
  const rows = text.split('\n').map((r) => r.replace(/\r$/, '')).filter((r) => r.length > 0);
  const W = Math.max(...rows.map((r) => r.length)) + 2;
  const H = rows.length + 2;
  const wall = new Uint8Array(W * H).fill(1);
  const goal = new Uint8Array(W * H);
  const box = new Uint8Array(W * H);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = (y + 1) * W + (x + 1);
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  });
  return { W, H, wall, goal, box, player };
}

// Connected floor components once the hallway row is cut out: each is a room.
function roomComponents(P, hrow) {
  const { W, H, wall } = P;
  const seen = new Uint8Array(W * H);
  const comps = [];
  for (let i = 0; i < W * H; i++) {
    if (wall[i] || seen[i] || Math.floor(i / W) === hrow) continue;
    const cells = [i];
    seen[i] = 1;
    for (let q = 0; q < cells.length; q++) {
      const c = cells[q];
      for (const d of [-W, W, -1, 1]) {
        const n = c + d;
        if (!wall[n] && !seen[n] && Math.floor(n / W) !== hrow) {
          seen[n] = 1;
          cells.push(n);
        }
      }
    }
    comps.push(cells);
  }
  return comps;
}

// Exact push-optimal search for one sub-board. inSub: squares the player may stand on; boxOK: squares a box may occupy.
// Returns the pushes as [box square, direction offset] in order, or null if the state budget runs out.
function solveSub(P, inSub, boxOK, boxes0, start) {
  const { W, H, goal } = P;
  const N = W * H;
  const dirs = [-W, W, -1, 1];

  // Dead squares: a box on a square outside `live` can never reach a goal, whatever the player does.
  const live = new Uint8Array(N);
  const queue = [];
  for (let i = 0; i < N; i++) {
    if (boxOK[i] && goal[i]) {
      live[i] = 1;
      queue.push(i);
    }
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const t = queue[qi];
    for (const d of dirs) {
      const s = t - d; // a box on s pushed by d lands on t, the player standing on s - d
      if (s < 0 || s >= N || !boxOK[s] || live[s] || !inSub[s - d]) continue;
      live[s] = 1;
      queue.push(s);
    }
  }

  const bx = new Uint8Array(N);
  const mark = new Int32Array(N);
  let stamp = 0;
  // Every square the player can reach from p without pushing, p included.
  function region(p) {
    stamp++;
    const q = [p];
    mark[p] = stamp;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      for (const d of dirs) {
        const n = c + d;
        if (inSub[n] && !bx[n] && mark[n] !== stamp) {
          mark[n] = stamp;
          q.push(n);
        }
      }
    }
    return q;
  }

  const boxes0s = boxes0.slice().sort((a, b) => a - b);
  bx.fill(0);
  for (const b of boxes0s) bx[b] = 1;
  const reg0 = region(start);
  const canon0 = Math.min(...reg0);
  const key0 = boxes0s.join(',') + '|' + canon0;
  // parent: state key -> { from, box, d } (null for the start). Layers are pushes, so the first goal met is optimal.
  const parent = new Map([[key0, null]]);
  let layer = [{ key: key0, boxes: boxes0s, player: start }];

  while (layer.length > 0) {
    const next = [];
    for (const st of layer) {
      if (st.boxes.every((b) => goal[b])) {
        const pushes = [];
        let info = parent.get(st.key);
        while (info) {
          pushes.push([info.box, info.d]);
          info = parent.get(info.from);
        }
        return pushes.reverse();
      }
      bx.fill(0);
      for (const b of st.boxes) bx[b] = 1;
      const reg = region(st.player);
      for (const c of reg) {
        for (const d of dirs) {
          const b = c + d;
          if (!bx[b]) continue;
          const t = b + d;
          if (!inSub[t] || !boxOK[t] || bx[t] || !live[t]) continue;
          bx[b] = 0;
          bx[t] = 1;
          const reg2 = region(b);
          bx[t] = 0;
          bx[b] = 1;
          let canon = reg2[0];
          for (const x of reg2) if (x < canon) canon = x;
          const nb = st.boxes.map((x) => (x === b ? t : x)).sort((x, y) => x - y);
          const key = nb.join(',') + '|' + canon;
          if (!parent.has(key)) {
            parent.set(key, { from: st.key, box: b, d });
            next.push({ key, boxes: nb, player: b });
          }
        }
      }
      if (parent.size > MAX_STATES) return null;
    }
    layer = next;
  }
  return null;
}

// Walk from `from` to `to` over open floor (no boxes, no walls); returns the step offsets, or null.
function walk(P, bxG, from, to) {
  if (from === to) return [];
  const N = P.W * P.H;
  const prev = new Int32Array(N).fill(-1);
  prev[from] = from;
  const q = [from];
  for (let qi = 0; qi < q.length && prev[to] === -1; qi++) {
    const c = q[qi];
    for (const d of [-P.W, P.W, -1, 1]) {
      const n = c + d;
      if (!P.wall[n] && !bxG[n] && prev[n] === -1) {
        prev[n] = c;
        q.push(n);
      }
    }
  }
  if (prev[to] === -1) return null;
  const steps = [];
  let c = to;
  while (c !== from) {
    const p = prev[c];
    steps.push(c - p);
    c = p;
  }
  return steps.reverse();
}

function letter(P, d) {
  if (d === -P.W) return 'u';
  if (d === P.W) return 'd';
  if (d === -1) return 'l';
  return 'r';
}

// One instance's moves in LURD. Returns '' when no complete solution is found (an empty answer is valid, and scores 0).
function solveInstance(text) {
  const P = parse(text);
  const N = P.W * P.H;
  const hrow = Math.floor(P.player / P.W);
  const isHall = (i) => !P.wall[i] && Math.floor(i / P.W) === hrow;

  // Rooms: each component of the floor with the hallway cut out. Structured only if every goal and box sits in a
  // room with as many goals as boxes, and each room opens onto the hallway.
  let structured = true;
  const subs = [];
  for (let i = 0; i < N && structured; i++) if (isHall(i) && (P.goal[i] || P.box[i])) structured = false;
  if (structured) {
    for (const cells of roomComponents(P, hrow)) {
      let nb = 0;
      let ng = 0;
      for (const c of cells) {
        nb += P.box[c];
        ng += P.goal[c];
      }
      if (nb === 0 && ng === 0) continue;
      if (nb !== ng || nb === 0) {
        structured = false;
        break;
      }
      const inSub = new Uint8Array(N);
      const boxOK = new Uint8Array(N);
      for (const c of cells) {
        inSub[c] = 1;
        boxOK[c] = 1;
      }
      let start = -1;
      for (const c of cells) {
        for (const n of [c - P.W, c + P.W]) {
          if (start < 0 && isHall(n)) start = n;
        }
      }
      for (let i = 0; i < N; i++) if (isHall(i)) inSub[i] = 1;
      if (start < 0) {
        structured = false;
        break;
      }
      const boxes = cells.filter((c) => P.box[c]);
      subs.push({ inSub, boxOK, boxes, start });
    }
  }
  if (!structured) {
    // Not the generator's shape: one search over the whole level.
    const inSub = new Uint8Array(N);
    for (let i = 0; i < N; i++) inSub[i] = P.wall[i] ? 0 : 1;
    subs.length = 0;
    subs.push({ inSub, boxOK: inSub, boxes: [...Array(N).keys()].filter((i) => P.box[i]), start: P.player });
  }

  // Solve each room; then replay all rooms' pushes on the real board, walking between them.
  const roomPushes = [];
  for (const s of subs) {
    const pushes = solveSub(P, s.inSub, s.boxOK, s.boxes, s.start);
    if (pushes === null) return '';
    roomPushes.push(pushes);
  }
  const bxG = P.box.slice();
  let player = P.player;
  const out = [];
  for (const pushes of roomPushes) {
    for (const [b, d] of pushes) {
      const path = walk(P, bxG, player, b - d);
      if (path === null) return '';
      for (const s of path) out.push(letter(P, s));
      out.push(letter(P, d));
      bxG[b] = 0;
      bxG[b + d] = 1;
      player = b;
    }
  }
  return out.join('');
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveInstance(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
