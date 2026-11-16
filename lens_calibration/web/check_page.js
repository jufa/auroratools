/* Checks the browser half of the alignment page against the Python half.

   Called by lens_calibration.selfcheck (Py -> node).  Two things are verified
   that Python cannot see from its side:

   1. forward() in the page -- the map that turns raster pixels into overlay
      pixels for the detections and the solved pairs -- reproduces
      session.Alignment.forward for a table of transforms computed in Python;
   2. the real event handlers turn a drag, the rotation slider, the mirror box
      and the arrow keys into exactly the alignment the solver is fed;
   3. the frame picker really posts the chosen file (from the file input and
      from a drop on the stage), a server with no frame leaves the stage as
      an invitation rather than an error, and the request that follows an open
      does not carry the time/place panel's old copy: doing so served the new
      frame with the previous frame's clock, so its star circles were drawn for
      the wrong sky;
   4. live pinning: a click on a circle picks that star, the second click posts
      the *photo* pixel under the pointer, the answer's transform is the one the
      photo gets, the residual line is drawn, pinning one star twice replaces
      rather than duplicates, undo refits through what is left, clicking the
      picked star's own circle places it instead of re-picking it, and a slow
      answer to an older click cannot land on top of a newer one;
   5. the drawn help stays help: the clusters, nebulae and Milky Way band are
      stroked and filled in their own colour from a payload key of their own,
      they share the star labels' collision rule, and -- because they are not in
      the list the hit test walks -- a click on one picks nothing.  A pin can
      also be taken out singly, by clicking the circle that carries it (live
      mode or not) or the ✕ in the readout, and the survivors are refitted.

   Usage: node check_page.js <index.html> <cases.json>
*/
"use strict";

const fs = require("fs");

const pagePath = process.argv[2];
const casesPath = process.argv[3];
if (!pagePath || !casesPath) {
  console.error("usage: node check_page.js <index.html> <cases.json>");
  process.exit(2);
}

const html = fs.readFileSync(pagePath, "utf8");
const open = html.indexOf("<script>");
const close = html.indexOf("</script>", open);
if (open < 0 || close < 0) {
  console.error("no <script> block found in " + pagePath);
  process.exit(2);
}
const source = html.slice(open + "<script>".length, close);
const spec = JSON.parse(fs.readFileSync(casesPath, "utf8"));

const results = [];
function check(name, ok, detail) {
  results.push(`${ok ? "ok  " : "FAIL"} ${name}${detail ? "  " + detail : ""}`);
}

/* ---- stub DOM: records listeners so the handlers can be fired by hand ---- */
function makeNode(id) {
  const node = {
    id,
    style: {},
    listeners: {},
    classList: {
      set: new Set(),
      add(name) { node.classList.set.add(name); },
      remove(name) { node.classList.set.delete(name); },
      contains(name) { return node.classList.set.has(name); },
      toggle(name, on) { (on ? node.classList.add : node.classList.remove)(name); },
    },
    value: "",
    textContent: "",
    innerHTML: "",
    checked: false,
    disabled: false,
    hidden: false,
    src: "",
    clicks: 0,
    addEventListener(type, handler) {
      (node.listeners[type] = node.listeners[type] || []).push(handler);
    },
    fire(type, event) {
      for (const handler of node.listeners[type] || []) {
        handler(Object.assign({ target: node, preventDefault() {} }, event));
      }
    },
    click() { node.clicks += 1; },
    removeAttribute(name) { if (name === "src") node.src = ""; },
    setPointerCapture() {},
    closest() { return null; },
    // The stage is at the origin of a 0,0 page, so client pixels are stage
    // pixels and the page's own scale is the only conversion left.
    getBoundingClientRect() {
      return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
    },
    getContext() {
      return {
        setTransform() {}, clearRect() {}, save() {}, restore() {}, beginPath() {},
        closePath() {}, arc() {}, moveTo() {}, lineTo() {}, stroke() {}, fill() {},
        fillText() {}, strokeText() {}, measureText() { return { width: 10 }; },
        setLineDash() {},
      };
    },
  };
  return node;
}

const nodes = {};
const documentStub = {
  getElementById: (id) => (nodes[id] = nodes[id] || makeNode(id)),
  activeElement: null,
};
const windowListeners = {};
const windowStub = {
  devicePixelRatio: 1,
  addEventListener(type, handler) {
    (windowListeners[type] = windowListeners[type] || []).push(handler);
  },
};

/* Node 14 has no FormData; the page only needs append() to carry the file. */
if (typeof globalThis.FormData !== "function") {
  globalThis.FormData = class FormData {
    constructor() { this.entries = []; }
    append(name, value, filename) {
      this.entries.push([name, (value && value.name) || filename || String(value)]);
    }
  };
}

/* fetch is recorded, never performed.  What matters is that the picker path
   puts the chosen file on the wire, and that a 409 from the server turns the
   stage into the invitation rather than into an error message. */
const calls = [];
let sessionReply = null;                 // null = offline
// /api/live answers by echoing the pairs it was sent (as the real endpoint
// does) and attaching the residual staged here for each star.
let liveReply = null;
// When liveDefer is on, /api/live answers are parked in heldLive instead of being
// delivered, so the test can hand them back out of order.
let liveDefer = false;
const heldLive = [];
// When liveHtml is on, /api/live answers with the HTML error page an older
// server (one without the endpoint) would send.
let liveHtml = false;
/* A reply as the browser hands it over: the status, and the body twice -- as
   text, which is what the page reads, and as JSON, which throws on an HTML page
   the way the real Response does. */
const stubResponse = (status, statusText, body) => ({
  status,
  ok: status >= 200 && status < 300,
  statusText,
  text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  json: async () => {
    if (typeof body === "string") {
      throw new SyntaxError(`Unexpected token '<', "${body.slice(0, 9)}..." is not valid JSON`);
    }
    return body;
  },
});
const reply = (status, body) => Promise.resolve(stubResponse(status, "", body));
const replyHtml = (status, statusText, html) => Promise.resolve(stubResponse(status, statusText, html));
const fetchStub = (url, init) => {
  calls.push({ url, method: (init && init.method) || "GET", body: init && init.body });
  if (url === "/api/open") {
    return reply(200, { name: "night.jpg", width: 1400, height: 1400 });
  }
  if (url.startsWith("/api/detections")) return reply(200, { stars: [], count: 0 });
  if (url === "/api/live") {
    if (liveHtml) {
      return replyHtml(404, "NOT FOUND",
        "<!doctype html>\n<html lang=en>\n<title>404 Not Found</title>\n<h1>Not Found</h1>");
    }
    if (!liveReply) return reply(400, { error: "no /api/live reply staged" });
    const residuals = liveReply.residuals || {};
    const sent = JSON.parse((init && init.body) || "{}");
    const draws = liveReply.draw || {};
    const answer = {
      alignment: liveReply.alignment,
      rms_px: liveReply.rms_px,
      max_px: liveReply.max_px,
      // From four pins the server also solves the lens through them and sends the
      // overlay re-projected through it; both are null below that count.
      lens: liveReply.lens || null,
      lens_error: liveReply.lens_error || null,
      overlay: liveReply.overlay || null,
      points: (sent.points || []).map((point) => {
        const base = {
          ...point,
          residual_px: point.name in residuals ? residuals[point.name] : 0,
        };
        const drawn = draws[point.name];
        return drawn ? { ...base, draw_x: drawn[0], draw_y: drawn[1] } : base;
      }),
    };
    if (!liveDefer) return reply(200, answer);
    return new Promise((resolve) => heldLive.push({ answer, resolve }));
  }
  if (sessionReply) return reply(sessionReply.status, sessionReply.body);
  return Promise.reject(new Error("offline"));
};
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

const page = new Function("window", "document", "fetch",
  source + "\nreturn { state, forward, inverse, draw, applyTransform, ringRadius, openFrame, loadSession,"
    + " sendPins, renderReadout, syncLive };" )(
  windowStub, documentStub, fetchStub);

/* ---- 1. the raster -> overlay map ---- */
page.state.session = { image: spec.image, overlay: null, detections: [] };
page.state.cssW = spec.stage_css.width;
page.state.cssH = spec.stage_css.height;
page.state.scale = spec.stage_css.scale;

let worst = 0;
let mismatches = 0;
for (const c of spec.transforms) {
  page.state.align = {
    roll_deg: c.roll_deg, mirrored: c.mirrored, dx: c.dx, dy: c.dy,
  };
  const got = page.forward(c.x, c.y);
  const err = Math.max(Math.abs(got.x - c.want_x), Math.abs(got.y - c.want_y));
  worst = Math.max(worst, err);
  if (err > 1e-9) {
    mismatches += 1;
    if (mismatches <= 5) {
      console.log(`    ${c.name}: js=(${got.x},${got.y}) py=(${c.want_x},${c.want_y})`);
    }
  }
}
check(`forward() matches Alignment.forward (${spec.transforms.length} cases, worst ${worst.toExponential(1)} px)`,
  mismatches === 0);

// The pointer arrives in stage pixels, the pin is posted in photo pixels, so the
// page needs the other direction too -- and it needs it to be *this* one: the
// inverse of the Python forward(), not a second-hand approximation of it.
let worstInverse = 0;
let inverseMismatches = 0;
for (const c of spec.transforms) {
  page.state.align = {
    roll_deg: c.roll_deg, mirrored: c.mirrored, dx: c.dx, dy: c.dy,
  };
  const got = page.inverse(c.want_x, c.want_y);
  const err = Math.max(Math.abs(got.x - c.x), Math.abs(got.y - c.y));
  worstInverse = Math.max(worstInverse, err);
  if (err > 1e-9) {
    inverseMismatches += 1;
    if (inverseMismatches <= 5) {
      console.log(`    ${c.name}: js=(${got.x},${got.y}) py=(${c.x},${c.y})`);
    }
  }
}
check(`inverse() undoes Alignment.forward (${spec.transforms.length} cases, worst ${worstInverse.toExponential(1)} px)`,
  inverseMismatches === 0);

/* ---- 2. the handlers ---- */
const scale = spec.stage_css.scale;
page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
const stage = documentStub.getElementById("stage");

stage.fire("pointerdown", { button: 0, pointerId: 1, clientX: 100, clientY: 100 });
stage.fire("pointermove", { pointerId: 1, clientX: 100 + 100 * scale, clientY: 100 - 50 * scale });
check("drag converts css pixels into raster pixels",
  Math.abs(page.state.align.dx - 100) < 1e-9 && Math.abs(page.state.align.dy + 50) < 1e-9,
  `dx=${page.state.align.dx} dy=${page.state.align.dy}`);
check("drag moves the photo with a css transform",
  documentStub.getElementById("frame").style.transform ===
    `translate(${(100 * scale).toFixed(2)}px, ${(-50 * scale).toFixed(2)}px) rotate(0deg)`,
  documentStub.getElementById("frame").style.transform);
stage.fire("pointermove", { pointerId: 1, clientX: 100 + 50 * scale, clientY: 100 });
check("the photo stays 1:1 with the grab point",
  Math.abs(page.state.align.dx - 50) < 1e-9 && Math.abs(page.state.align.dy) < 1e-9,
  `dx=${page.state.align.dx} dy=${page.state.align.dy}`);
stage.fire("pointerup", { pointerId: 1 });
stage.fire("pointermove", { pointerId: 1, clientX: 100 + 400 * scale, clientY: 400 });
check("nothing follows the pointer after pointerup",
  Math.abs(page.state.align.dx - 50) < 1e-9, `dx=${page.state.align.dx}`);

documentStub.getElementById("roll").fire("input", { target: { value: "-119.3" } });
check("the rotation slider sets the roll",
  Math.abs(page.state.align.roll_deg + 119.3) < 1e-9, `roll=${page.state.align.roll_deg}`);
check("the transform carries the roll",
  documentStub.getElementById("frame").style.transform.endsWith("rotate(-119.3deg)"),
  documentStub.getElementById("frame").style.transform);

documentStub.getElementById("mirror").fire("change", { target: { checked: true } });
check("the mirror box flips the photo",
  page.state.align.mirrored === true &&
    documentStub.getElementById("frame").style.transform.endsWith("scaleX(-1)"),
  documentStub.getElementById("frame").style.transform);

const dxKey = page.state.align.dx;
const dyKey = page.state.align.dy;
const fireKey = (key, shiftKey, target) => {
  for (const handler of windowListeners.keydown || []) {
    handler({ key, shiftKey, target: target || documentStub.getElementById("stage"), preventDefault() {} });
  }
};
fireKey("ArrowLeft", false);
check("an arrow key nudges 1 px", Math.abs(page.state.align.dx - (dxKey - 1)) < 1e-9,
  `dx=${page.state.align.dx}`);
fireKey("ArrowDown", true);
check("shift + arrow nudges 10 px", Math.abs(page.state.align.dy - (dyKey + 10)) < 1e-9,
  `dy=${page.state.align.dy}`);
let selector = null;
fireKey("ArrowLeft", false, { closest: (sel) => { selector = sel; return {}; } });
check("keys are ignored while typing in a field",
  selector === "input, select, textarea" && Math.abs(page.state.align.dx - (dxKey - 1)) < 1e-9,
  `selector=${selector} dx=${page.state.align.dx}`);

documentStub.getElementById("reset").fire("click", {});
check("reset clears the alignment",
  page.state.align.roll_deg === 0 && page.state.align.mirrored === false &&
    page.state.align.dx === 0 && page.state.align.dy === 0);

/* ---- 3. the draw path ---- */
const counts = { arc: 0, strokeText: 0, fillText: 0, moveTo: 0, lineTo: 0, fill: 0 };
const countingCtx = {
  setTransform() {}, clearRect() {}, save() {}, restore() {}, beginPath() {},
  closePath() {},
  moveTo() { counts.moveTo += 1; }, lineTo() { counts.lineTo += 1; },
  stroke() {}, fill() { counts.fill += 1; }, setLineDash() {},
  arc() { counts.arc += 1; },
  fillText() { counts.fillText += 1; },
  strokeText() { counts.strokeText += 1; },
  measureText() { return { width: 10 }; },
};
const overlay = {
  projection: "equidistant",
  fov_deg: 180,
  theta_edge_deg: 90,
  radius_px: 691.4,
  centre: { x: 700, y: 700 },
  grid: {
    rings: [
      { theta_deg: 30, radius_px: 230.5 },
      { theta_deg: 60, radius_px: 460.9 },
      { theta_deg: 90, radius_px: 691.4 },
    ],
    meridians: [
      { az_deg: 0, label: "N", x: 700, y: 8.6 },
      { az_deg: 90, label: "E", x: 8.6, y: 700 },
      { az_deg: 180, label: "S", x: 700, y: 1391.4 },
      { az_deg: 270, label: "W", x: 1391.4, y: 700 },
    ],
  },
  stars: [],
};
for (let i = 0; i < 20; i += 1) {
  overlay.stars.push({ name: "s" + i, vmag: 1 + (i % 4), x: 300 + 9 * i, y: 400 + 7 * i });
}
page.state.session.overlay = overlay;
page.state.ctx = countingCtx;
page.state.cssW = 700;
page.state.cssH = 700;
page.state.scale = 0.5;

const stars = overlay.stars.slice();
const drawAndCount = (showGrid, starList) => {
  overlay.stars = starList;
  page.state.showGrid = showGrid;
  counts.arc = 0;
  counts.strokeText = 0;
  counts.fillText = 0;
  counts.moveTo = 0;
  counts.lineTo = 0;
  counts.fill = 0;
  page.draw();
  return {
    arc: counts.arc, strokeText: counts.strokeText, fillText: counts.fillText,
    moveTo: counts.moveTo, lineTo: counts.lineTo, fill: counts.fill,
  };
};

const gridOn = drawAndCount(true, stars);
const gridOff = drawAndCount(false, stars);
const bare = drawAndCount(true, []);
const rings = overlay.grid.rings.length;
const meridians = overlay.grid.meridians.length;

// the rim and every grid ring are stroked twice: a dark halo, then the line
check("draw() strokes every grid ring (halo + line)",
  gridOn.arc - gridOff.arc === 2 * rings, `on=${gridOn.arc} off=${gridOff.arc}`);
check("draw() strokes a rim plus one circle per star",
  gridOff.arc === 2 + stars.length, `arcs=${gridOff.arc}`);
check("draw() labels each cardinal point once",
  bare.strokeText === meridians, `labels=${bare.strokeText}`);
check("draw() drops colliding star labels instead of piling them up",
  gridOn.fillText >= meridians && gridOn.fillText < meridians + stars.length,
  `labels=${gridOn.fillText} of ${meridians + stars.length}`);

/* ---- 3b. the aid layer: drawn, and impossible to pin ---- */
/* Clusters, nebulae and the Milky Way are orientation cues, not control points:
   they are drawn in their own colour from a payload key of their own, their
   names take part in the same collision rule as the star names, and they are
   deliberately absent from overlay.stars -- which is the list starAt() walks, so
   nothing here can ever be picked as a pin. */
overlay.aids = {
  objects: [
    { name: "Pleiades", kind: "cluster", radius_px: 14, x: 500, y: 500 },
    { name: "LMC", kind: "galaxy", radius_px: 40, x: 900, y: 700 },
  ],
  band: [[[100, 100], [200, 120], [210, 160], [110, 150]]],
};
page.state.showAids = true;
const aidsOn = drawAndCount(false, []);
page.state.showAids = false;
const aidsOff = drawAndCount(false, []);
overlay.stars = stars;
check("draw() strokes one circle per sky aid and nothing when the box is off",
  aidsOn.arc - aidsOff.arc === overlay.aids.objects.length,
  `on=${aidsOn.arc} off=${aidsOff.arc}`);
check("draw() fills the Milky Way band, one fill per polygon",
  aidsOn.fill - aidsOff.fill === overlay.aids.band.length,
  `on=${aidsOn.fill} off=${aidsOff.fill}`);
check("aid names are labelled under the same collision rule as star names",
  aidsOn.fillText - aidsOff.fillText === overlay.aids.objects.length,
  `labels=${aidsOn.fillText - aidsOff.fillText}`);
page.state.showAids = true;

/* ---- 4. the frame picker ---- */
/* The real handlers are fired here too: the point of the picker is that the
   file the user chose is the file the server reads, and that an empty server
   shows an invitation rather than an error. */
(async () => {
  const sessionPayload = {
    image: { url: "/frame.jpg", name: "night.jpg", width: 1400, height: 1400 },
    meta: { camera: "SONY ZV-E10", lens: "Meike 6.5 mm", exposure_s: 8, iso: 800 },
    time: {
      captured_utc: "2026-01-18T09:15:17", latitude: 64.5724, longitude: -149.262,
      source: "EXIF", position_source: "EXIF",
    },
    circle: {
      xc_frac: 0.5, yc_frac: 0.5, sigma_edge: 0.98774, contrast: 20.4,
      confidence: 0.93, axis_ratio: 1.001,
    },
    overlay: null,
    overlay_error: "no time or place",
    alignment: {},
    seed_camera: null,
    transform_check_px: null,
    counts: { detections: 0, point_like: 0, catalogue: 0 },
    projections: ["equidistant", "equisolid"],
    warnings: [],
    calibration_path: "/tmp/night.json",
  };
  const stageNode = documentStub.getElementById("stage");
  const input = documentStub.getElementById("openFile");
  const frameName = () => documentStub.getElementById("frameName").textContent;

  sessionReply = { status: 409, body: { error: "no frame is open -- choose an image first" } };
  await page.loadSession();
  check("a server with no frame puts the stage in its empty state",
    page.state.session === null && stageNode.classList.contains("empty") &&
      frameName() === "no frame",
    `frame=${frameName()} empty=${stageNode.classList.contains("empty")}`);
  check("the empty state says how to open a frame",
    documentStub.getElementById("readout").innerHTML.includes("Open image"),
    documentStub.getElementById("readout").innerHTML.slice(0, 50));

  sessionReply = { status: 200, body: sessionPayload };
  input.files = [{ name: "night.jpg" }];
  input.value = "/home/user/night.jpg";     // what a real file input carries
  input.fire("change", {});
  await settle();
  await settle();
  const opened = calls.find((call) => call.url === "/api/open");
  check("choosing a file posts it to /api/open",
    Boolean(opened) && opened.method === "POST" &&
      opened.body.entries.some(([name, value]) => name === "file" && value === "night.jpg"),
    opened ? JSON.stringify(opened.body.entries) : "no /api/open call");
  check("the file input clears itself so the same file can be chosen again",
    input.value === "", `value=${JSON.stringify(input.value)}`);
  check("the opened frame replaces the empty state",
    page.state.session !== null && !stageNode.classList.contains("empty") &&
      frameName() === "night.jpg",
    `frame=${frameName()} empty=${stageNode.classList.contains("empty")}`);
  // The status line is the only place the user looks, so a missing overlay has
  // to say *why* there, not just that it is missing.
  check("a frame without an overlay says why in the status bar",
    documentStub.getElementById("status").textContent ===
      "overlay unavailable: no time or place",
    documentStub.getElementById("status").textContent);

  page.state.session = null;
  input.clicks = 0;
  stageNode.fire("click", {});
  check("clicking the empty stage opens the picker", input.clicks === 1,
    `clicks=${input.clicks}`);

  const before = calls.length;
  stageNode.fire("drop", {
    dataTransfer: { types: ["Files"], files: [{ name: "dropped.jpg" }] },
  });
  await settle();
  await settle();
  const dropped = calls.slice(before).find((call) => call.url === "/api/open");
  check("dropping a file on the stage opens it",
    Boolean(dropped) &&
      dropped.body.entries.some(([name, value]) => name === "file" && value === "dropped.jpg"),
    dropped ? JSON.stringify(dropped.body.entries) : "no /api/open call");

  /* The time/place panel *shows* what is in effect; it must not be sent back
     with every request.  Echoing it carried the previous frame's clock and
     position onto the frame just opened, so the fresh frame got the old sky's
     star circles and they never moved. */
  const sessionCalls = () => calls.filter((call) => call.url.startsWith("/api/session"));
  const queryOf = (call) => new URLSearchParams(call.url.split("?")[1] || "");

  page.state.session = null;
  input.files = [{ name: "second.jpg" }];
  input.value = "/home/user/second.jpg";
  input.fire("change", {});
  await settle();
  await settle();
  const asked = sessionCalls().slice(-1)[0];
  const askedQuery = asked ? queryOf(asked) : new URLSearchParams();
  check("opening a frame asks for its own time and place, not the panel's",
    Boolean(asked) && !askedQuery.has("utc") && !askedQuery.has("lat") &&
      !askedQuery.has("lon"),
    asked ? asked.url : "no /api/session call");

  documentStub.getElementById("utc").value = "2026-01-19T03:00";
  documentStub.getElementById("lat").value = "64.6";
  documentStub.getElementById("lon").value = "-149.3";
  documentStub.getElementById("applySite").fire("click", {});
  await settle();
  await settle();
  const applied = queryOf(sessionCalls().slice(-1)[0]);
  check("apply time / place sends exactly what was typed",
    applied.get("utc") === "2026-01-19T03:00:00" && applied.get("lat") === "64.6" &&
      applied.get("lon") === "-149.3",
    sessionCalls().slice(-1)[0].url);

  sessionReply = { status: 409, body: { error: "no frame is open -- choose an image first" } };
  await page.loadSession();
  check("the empty state drops the old frame's time and place",
    documentStub.getElementById("utc").value === "" &&
      documentStub.getElementById("lat").value === "" &&
      documentStub.getElementById("lon").value === "",
    JSON.stringify([documentStub.getElementById("utc").value,
                    documentStub.getElementById("lat").value]));

  /* ---- 5. live alignment: pin a star, refit through every pair ---- */
  /* The workflow the user drives: click a circle, then click the same star in
     the photo.  What matters is that the click reaches the server as a *pair*
     -- the circle's place in the sky and the pixel in the photo -- that the
     transform it answers with is the one the page then uses, and that every
     pair is drawn as a line from the photo to the sky, which is the only
     feedback that says how well the pins agree. */
  const liveStars = [
    { name: "Vega", vmag: 0.1, x: 300, y: 400 },
    { name: "Altair", vmag: 0.9, x: 900, y: 420 },
    { name: "Deneb", vmag: 1.3, x: 620, y: 1000 },
  ];
  page.state.session = {
    ...sessionPayload,
    overlay: {
      projection: "equidistant",
      fov_deg: 180,
      theta_edge_deg: 90,
      radius_px: 691.4,
      centre: { x: 700, y: 700 },
      count: liveStars.length,
      mag_limit: 5.5,
      grid: { rings: [], meridians: [] },
      stars: liveStars,
      // One aid, well away from every ring: drawn, but never a control point.
      aids: {
        objects: [{ name: "Pleiades", kind: "cluster", radius_px: 14, x: 1050, y: 900 }],
        band: [],
      },
    },
    seed_camera: { roll_deg: 0, centre_px: { x: 700, y: 700 }, mirrored: false },
    transform_check_px: 0,
  };
  page.state.detections = [];
  page.state.ctx = countingCtx;
  page.state.cssW = 700;
  page.state.cssH = 700;
  page.state.scale = 0.5;
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  page.state.live = { on: false, pending: null, points: [], rms: null, max: null };
  page.state.showGrid = false;
  page.state.showDetections = false;

  const liveStage = documentStub.getElementById("stage");
  const liveNote = () => documentStub.getElementById("liveNote").textContent;
  const liveButton = documentStub.getElementById("live");

  // The stage is the raster at half size: a css point is half a raster pixel.
  const clickAt = (x, y) => {
    liveStage.fire("pointerdown", { button: 0, pointerId: 7, clientX: x, clientY: y });
    liveStage.fire("pointerup", { pointerId: 7, clientX: x, clientY: y });
  };
  const lastPin = () => calls.filter((call) => call.url === "/api/live").slice(-1)[0];
  /* A circle that carries a pin is the removal handle, so a block that starts by
     clicking Vega's circle has to start from a pin set without Vega in it --
     otherwise the click takes that pin out instead of picking the star. */
  const forgetPins = () => {
    page.state.live.points = [];
    page.state.live.pending = null;
    page.syncLive();
  };

  check("live alignment waits to be asked for",
    page.state.live.on === false && !liveStage.classList.contains("live"));
  liveButton.fire("click", {});
  check("asking for it turns the stage into a picker",
    page.state.live.on === true && liveStage.classList.contains("live") &&
      liveButton.textContent === "done",
    `on=${page.state.live.on} button=${liveButton.textContent}`);

  // A sky aid is drawn on the overlay but is not a control point: the hit test
  // walks overlay.stars only, so clicking the Pleiades picks nothing and posts
  // nothing.  That is what keeps "visual help" from quietly becoming data.
  const beforeAidClick = calls.length;
  clickAt(1050 * 0.5, 900 * 0.5);
  check("clicking a sky aid picks no star and posts no pin",
    page.state.live.pending === null && page.state.live.points.length === 0 &&
      calls.slice(beforeAidClick).filter((call) => call.url === "/api/live").length === 0,
    `pending=${JSON.stringify(page.state.live.pending)}`);

  clickAt(300 * 0.5, 400 * 0.5);                  // on Vega's circle
  check("a click on a circle picks that star",
    Boolean(page.state.live.pending) && page.state.live.pending.name === "Vega" &&
      page.state.live.pending.star_x === 300 && page.state.live.pending.star_y === 400,
    JSON.stringify(page.state.live.pending));
  check("the panel then asks for the star in the photo",
    liveNote().includes("Vega"), liveNote());

  liveReply = {
    alignment: { roll_deg: -12.5, mirrored: false, dx: 30, dy: -8 },
    residuals: { Vega: 3.5 },
    rms_px: 3.5,
    max_px: 3.5,
  };
  const beforePin = calls.length;
  clickAt(250, 150);                              // (500, 300) in raster pixels
  await settle();
  await settle();
  const posted = lastPin();
  const pinBody = posted && posted.body ? JSON.parse(posted.body) : { points: [] };
  check("the click posts the pair in both frames",
    calls.slice(beforePin).filter((call) => call.url === "/api/live").length === 1 &&
      posted.method === "POST" &&
      pinBody.points.length === 1 &&
      pinBody.points[0].name === "Vega" &&
      pinBody.points[0].star_x === 300 && pinBody.points[0].star_y === 400 &&
      pinBody.points[0].x === 500 && pinBody.points[0].y === 300,
    JSON.stringify(pinBody.points));
  check("the transform the refit answers with is the one the photo gets",
    page.state.align.roll_deg === -12.5 && page.state.align.dx === 30 &&
      page.state.align.dy === -8 &&
      documentStub.getElementById("frame").style.transform.endsWith("rotate(-12.5deg)"),
    documentStub.getElementById("frame").style.transform);
  check("the pin is kept, with the error it still carries",
    page.state.live.points.length === 1 && page.state.live.points[0].residual_px === 3.5 &&
      page.state.live.rms === 3.5 &&
      documentStub.getElementById("readout").innerHTML.includes("Pinned"),
    `points=${page.state.live.points.length} rms=${page.state.live.rms}`);
  check("a refit invalidates the solve that described the old transform",
    page.state.solved === null && documentStub.getElementById("save").disabled === true);

  // moveTo/lineTo per pin: one dashed line out to the circle, then a crosshair
  // on the star where the user put it -- three moves and three lines in total.
  const drawFootAndHead = () => {
    counts.moveTo = 0;
    counts.lineTo = 0;
    page.draw();
    return { moveTo: counts.moveTo, lineTo: counts.lineTo };
  };
  const savedPoints = page.state.live.points;
  page.state.live.points = [];
  const bare = drawFootAndHead();
  page.state.live.points = savedPoints;
  const drawn = drawFootAndHead();
  check("every pin is drawn from the photo to its circle",
    drawn.lineTo - bare.lineTo === 3 && drawn.moveTo - bare.moveTo === 3,
    `lines=${drawn.lineTo - bare.lineTo} moves=${drawn.moveTo - bare.moveTo}`);

  liveReply = {
    alignment: { roll_deg: -10, mirrored: false, dx: 20, dy: -5 },
    residuals: { Vega: 2.75, Altair: 1.25 },
    rms_px: 2.15,
    max_px: 2.75,
  };
  clickAt(900 * 0.5, 420 * 0.5);                  // Altair's circle
  clickAt(445, 215);                              // (890, 430) in raster pixels
  await settle();
  await settle();
  const second = JSON.parse(lastPin().body);
  check("a second pin is fitted together with the first, not instead of it",
    second.points.length === 2 &&
      second.points.map((point) => point.name).join(",") === "Vega,Altair",
    JSON.stringify(second.points.map((point) => [point.name, point.x, point.y])));

  clickAt(620 * 0.5, 1000 * 0.5);                 // pick Deneb, then change our mind
  fireKey("Escape", false);
  check("esc drops the star that was waiting to be placed",
    page.state.live.pending === null && page.state.live.on === true &&
      page.state.live.points.length === 2,
    `pending=${JSON.stringify(page.state.live.pending)}`);
  fireKey("Escape", false);
  check("esc again leaves live alignment, keeping the pins",
    page.state.live.on === false && !liveStage.classList.contains("live") &&
      page.state.live.points.length === 2,
    `on=${page.state.live.on}`);

  const beforeUndo = calls.length;
  documentStub.getElementById("liveUndo").fire("click", {});
  await settle();
  await settle();
  const undone = calls.slice(beforeUndo).filter((call) => call.url === "/api/live");
  check("undo refits through the pins that are left",
    undone.length === 1 && JSON.parse(undone[0].body).points.length === 1 &&
      page.state.live.points.length === 1 && page.state.live.points[0].name === "Vega",
    `calls=${undone.length} points=${page.state.live.points.length}`);

  const beforeLast = calls.length;
  documentStub.getElementById("liveUndo").fire("click", {});
  await settle();
  await settle();
  check("undoing the last pin needs no refit at all",
    calls.slice(beforeLast).filter((call) => call.url === "/api/live").length === 0 &&
      page.state.live.points.length === 0 && page.state.live.rms === null,
    `points=${page.state.live.points.length} rms=${page.state.live.rms}`);

  /* ---- 5c. taking one pin out, from its circle or from the list ---- */
  /* A pin that turns out to be wrong is far more common than a pin that turns
     out to be the last one, so a single pair has to be droppable on its own:
     clicking the circle that carries it, or the ✕ beside it in the readout.
     Both refit through what is left -- the server never nudges an existing fit,
     so the removed pair leaves nothing behind -- and with nothing left nothing
     is posted at all, which is the page's own "forget them". */
  const pinSet = [
    { name: "Vega", star_x: 300, star_y: 400, x: 500, y: 300, residual_px: 2 },
    { name: "Altair", star_x: 900, star_y: 420, x: 890, y: 430, residual_px: 1.4 },
    { name: "Deneb", star_x: 620, star_y: 1000, x: 610, y: 990, residual_px: 1.1 },
  ];
  page.state.live.points = pinSet;
  page.state.live.on = false;             // the pins outlive the mode
  page.syncLive();
  liveReply = {
    alignment: { roll_deg: -11, mirrored: false, dx: 18, dy: -4 },
    residuals: { Vega: 1.2, Deneb: 0.9 },
    rms_px: 1.1,
    max_px: 1.2,
  };
  const beforeCircleDrop = calls.length;
  clickAt(900 * 0.5, 420 * 0.5);          // Altair's circle, live mode off
  await settle();
  await settle();
  const circleDrop = calls.slice(beforeCircleDrop).filter((call) => call.url === "/api/live");
  check("clicking a pinned circle takes that pin out, live mode or not",
    circleDrop.length === 1 &&
      JSON.parse(circleDrop[0].body).points.map((point) => point.name).join(",") ===
        "Vega,Deneb" &&
      page.state.live.points.map((point) => point.name).join(",") === "Vega,Deneb",
    `calls=${circleDrop.length} left=${page.state.live.points.map((p) => p.name).join(",")}`);

  const beforeListDrop = calls.length;
  documentStub.getElementById("readout").fire("click", {
    target: { dataset: { remove: "Deneb" } },
  });
  await settle();
  await settle();
  const listDrop = calls.slice(beforeListDrop).filter((call) => call.url === "/api/live");
  check("the ✕ in the pin list removes that pin and refits through the rest",
    listDrop.length === 1 &&
      JSON.parse(listDrop[0].body).points.map((point) => point.name).join(",") === "Vega" &&
      page.state.live.points.length === 1,
    `calls=${listDrop.length} left=${page.state.live.points.map((p) => p.name).join(",")}`);

  const beforeEmpty = calls.length;
  documentStub.getElementById("readout").fire("click", {
    target: { dataset: { remove: "Vega" } },
  });
  await settle();
  await settle();
  check("removing the last pin needs no refit and clears the lens",
    calls.slice(beforeEmpty).filter((call) => call.url === "/api/live").length === 0 &&
      page.state.live.points.length === 0 && page.state.live.lens === null &&
      page.state.live.overlay === null && !page.state.live.pending,
    `points=${page.state.live.points.length} lens=${page.state.live.lens}`);

  /* ---- 5b. the fourth pin hands over to the lens ---- */
  /* Four pins are the point where the server stops moving the photo and starts
     solving the fisheye: it answers with the centre, focal length and k terms,
     plus the overlay re-projected through them.  The page has to draw that
     overlay (rings on the calibrated positions) while still remembering each
     ring's *ideal* position, because that is what a pin's field angle is read
     from -- posting the drawn position instead would make the fifth pin feed the
     fit its own output. */
  const threePins = [
    { name: "Vega", star_x: 300, star_y: 400, x: 500, y: 300, residual_px: 2 },
    { name: "Altair", star_x: 900, star_y: 420, x: 890, y: 430, residual_px: 2 },
    { name: "Deneb", star_x: 620, star_y: 1000, x: 610, y: 990, residual_px: 2 },
  ];
  const fourPins = threePins.concat([
    { name: "Polaris", star_x: 1000, star_y: 1050, x: 990, y: 1040, residual_px: 1.1 },
  ]);
  page.state.live.points = threePins;
  page.state.live.on = true;
  page.syncLive();
  check("three pins say one more star measures the lens",
    liveNote().includes("one more star solves the lens"), liveNote());

  const calibrated = liveStars.map((star) => ({
    ...star, draw_x: star.x + 6, draw_y: star.y - 4,
  }));
  liveReply = {
    alignment: { roll_deg: -10, mirrored: false, dx: 20, dy: -5 },
    residuals: { Vega: 0.9, Altair: 1.4, Deneb: 2.1 },
    draw: { Vega: [306, 396], Altair: [906, 416], Deneb: [626, 996] },
    rms_px: 1.5,
    max_px: 2.1,
    lens: {
      pins: 4, num_coefficients: 1, focal_px: 512.3, coefficients: [-0.0538],
      centre_px: { x: 678.5, y: 707.9 }, covers_sigma: 0.85,
      lens: { theta_sigma_edge_deg: 88.62 }, notes: ["pin a fifth star"],
    },
    overlay: {
      projection: "equidistant", fov_deg: 180, calibrated: true,
      theta_edge_deg: 88.62, radius_px: 691.4, centre: { x: 678.5, y: 707.9 },
      count: calibrated.length, mag_limit: 5.5,
      grid: { rings: [], meridians: [] }, stars: calibrated,
    },
  };
  await page.sendPins(fourPins);
  await settle();
  check("four pins measure the lens and the page keeps it",
    Boolean(page.state.live.lens) && page.state.live.lens.num_coefficients === 1 &&
      page.state.live.overlay !== null && page.state.live.overlay.calibrated === true &&
      liveNote().includes("1 k term"),
    `lens=${JSON.stringify(page.state.live.lens && page.state.live.lens.focal_px)} note=${liveNote()}`);
  check("the readout shows what the pins measured",
    documentStub.getElementById("readout").innerHTML.includes("Focal length") &&
      documentStub.getElementById("readout").innerHTML.includes("512.3") &&
      documentStub.getElementById("readout").innerHTML.includes("pin a fifth star"),
    documentStub.getElementById("readout").innerHTML.slice(0, 120));

  // The rings are drawn where the calibration puts them, not where the ideal sky
  // did: sample the arcs the draw pass asks for.
  const arcs = [];
  const recordingCtx = Object.create(countingCtx);
  recordingCtx.arc = (x, y, r) => { arcs.push([x, y]); };
  const previousCtx = page.state.ctx;
  page.state.ctx = recordingCtx;
  page.draw();
  page.state.ctx = previousCtx;
  const drawnAt = (x, y) => arcs.some(([ax, ay]) => Math.abs(ax - x * 0.5) < 0.01 && Math.abs(ay - y * 0.5) < 0.01);
  check("the circles move onto the calibrated positions",
    drawnAt(306, 396) && !drawnAt(300, 400), JSON.stringify(arcs.slice(-3)));

  // A pin taken against the calibrated overlay still posts the *ideal* position:
  // the ring is drawn where the calibration puts it, but the pair the server
  // needs carries the sky position the field angle is read from -- posting the
  // drawn one would feed the fit its own output and the fifth pin would say
  // nothing new.
  page.state.live.on = true;
  page.state.live.pending = null;
  // Vega carries a pin from the fourth star, and a pinned circle is now the
  // removal handle: drop its pair first so the click below picks the star again,
  // which is also how a pin that landed on the wrong speck is corrected.
  page.state.live.points = fourPins.filter((point) => point.name !== "Vega");
  page.syncLive();
  const beforeFifth = calls.length;
  clickAt(306 * 0.5, 396 * 0.5);                  // the *drawn* Vega circle
  const picked = page.state.live.pending;
  clickAt(250, 150);                              // (500, 300) in the photo
  await settle();
  await settle();
  const fifth = calls.slice(beforeFifth).filter((call) => call.url === "/api/live");
  const fifthBody = fifth.length ? JSON.parse(fifth[0].body) : { points: [] };
  check("a pin on the calibrated overlay still posts the ideal position",
    Boolean(picked) && picked.name === "Vega" && picked.star_x === 300 && picked.star_y === 400 &&
      fifthBody.points.length === 4 &&
      fifthBody.points.some((point) => point.name === "Vega" && point.star_x === 300 && point.star_y === 400),
    `picked=${JSON.stringify(picked)} posted=${JSON.stringify(fifthBody.points.map((p) => [p.name, p.star_x]))}`);

  // The series stops at theta^5.  A sixth star is another equation for the same
  // five unknowns, and the readout repeats the server's note about it rather
  // than promising a coefficient that will never appear.
  const sixPins = fourPins.concat([
    { name: "Sirius", star_x: 620, star_y: 300, x: 615, y: 305, residual_px: 0.8 },
  ]);
  const capped = {
    pins: 6, num_coefficients: 2, focal_px: 512.3, coefficients: [-0.0538, 0.0021],
    centre_px: { x: 678.5, y: 707.9 }, covers_sigma: 0.85,
    lens: { theta_sigma_edge_deg: 88.62 },
    notes: ["6 pins, but the fit stops at the theta^5 term: the stars past the fifth"
      + " over-determine the same five unknowns rather than adding coefficients"],
  };
  liveReply = {
    alignment: { roll_deg: -10, mirrored: false, dx: 20, dy: -5 },
    residuals: { Vega: 0.9, Altair: 1.4, Deneb: 2.1, Polaris: 1.1, Sirius: 0.8 },
    rms_px: 1.4,
    max_px: 2.1,
    lens: capped,
    overlay: {
      projection: "equidistant", fov_deg: 180, calibrated: true,
      theta_edge_deg: 88.62, radius_px: 691.4, centre: { x: 678.5, y: 707.9 },
      count: calibrated.length, mag_limit: 5.5,
      grid: { rings: [], meridians: [] }, stars: calibrated,
    },
  };
  await page.sendPins(sixPins);
  await settle();
  check("a sixth star adds equations, not a coefficient",
    Boolean(page.state.live.lens) && page.state.live.lens.num_coefficients === 2 &&
      liveNote().includes("2 k terms") &&
      documentStub.getElementById("readout").innerHTML.includes("over-determine"),
    `note=${liveNote()} readout=${documentStub.getElementById("readout").innerHTML.slice(0, 160)}`);

  // Dropping back to three pins takes the lens away again, so the page cannot go
  // on displaying a calibration its pins no longer support.
  liveReply = {
    alignment: { roll_deg: -10, mirrored: false, dx: 20, dy: -5 },
    residuals: { Vega: 2, Altair: 2, Deneb: 2 },
    rms_px: 2,
    max_px: 2,
  };
  await page.sendPins(threePins);
  await settle();
  check("three pins take the lens away again",
    page.state.live.lens === null && page.state.live.overlay === null &&
      !documentStub.getElementById("readout").innerHTML.includes("Focal length"),
    `lens=${JSON.stringify(page.state.live.lens)}`);

  // A minimum-pin fit can legitimately fail (a star pinned on the overlay centre
  // has no field angle, for instance).  The answer then carries the reason
  // *beside* the refitted alignment instead of replacing it, and the page has to
  // show it rather than looking like the fourth pin did nothing.
  liveReply = {
    alignment: { roll_deg: -10, mirrored: false, dx: 20, dy: -5 },
    residuals: { Vega: 2, Altair: 2, Deneb: 2, Polaris: 2 },
    rms_px: 2,
    max_px: 2,
    lens_error: "a pinned star sits on the overlay centre: pin a star away from the middle",
  };
  await page.sendPins(fourPins);
  await settle();
  check("a lens the pins cannot support is reported, not swallowed",
    page.state.live.lens === null && page.state.live.lensError !== null &&
      documentStub.getElementById("readout").innerHTML.includes("overlay centre") &&
      liveNote().includes("overlay centre"),
    liveNote());

  // Hand the pin state back the way the next blocks expect it (no pins, live
  // off), so this section cannot leak into the ones after it.
  page.state.live = {
    on: false, pending: null, points: [], rms: null, max: null,
    lens: null, lensError: null, overlay: null,
  };

  // The one dead end this section guards against: with a star picked, clicking
  // its own circle must place the pin -- not pick the star again -- or a star
  // that is already where the overlay says it is could never be pinned at all.
  // The photo is standing still here, so stage and photo pixels are the same.
  liveButton.fire("click", {});
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  liveReply = {
    alignment: { roll_deg: 0, mirrored: false, dx: 0, dy: 0 },
    residuals: { Vega: 0 },
    rms_px: 0,
    max_px: 0,
  };
  clickAt(300 * 0.5, 400 * 0.5);                  // Vega's circle
  const beforeOwn = calls.length;
  clickAt(300 * 0.5, 400 * 0.5);                  // the same circle again
  await settle();
  await settle();
  const own = calls.slice(beforeOwn).filter((call) => call.url === "/api/live");
  check("clicking the picked star's own circle places it there",
    own.length === 1 && JSON.parse(own[0].body).points[0].name === "Vega" &&
      JSON.parse(own[0].body).points[0].x === 300 && page.state.live.pending === null,
    `calls=${own.length} pending=${JSON.stringify(page.state.live.pending)}`);

  // The pin has to be a pixel of the *photograph*, not of the stage.  Once the
  // photo is turned, the two disagree: with a quarter turn about the centre the
  // photo pixel (700, 1100) is drawn at stage (300, 700), so a click there must
  // be posted as 700, 1100 -- posting the stage pixel instead would make every
  // refit turn the photo by the same angle again and the pins would walk away.
  page.state.align = { roll_deg: 90, mirrored: false, dx: 0, dy: 0 };
  forgetPins();                                   // Vega carries the pin above
  liveReply = {
    alignment: { roll_deg: 90, mirrored: false, dx: 0, dy: 0 },
    residuals: { Vega: 0 },
    rms_px: 0,
    max_px: 0,
  };
  clickAt(300 * 0.5, 400 * 0.5);                  // Vega's circle, in stage pixels
  const seen = page.forward(700, 1100);           // where the photo puts that pixel
  const beforeTurned = calls.length;
  clickAt(seen.x * 0.5, seen.y * 0.5);
  await settle();
  await settle();
  const turned = calls.slice(beforeTurned).filter((call) => call.url === "/api/live");
  const turnedPin = turned.length ? JSON.parse(turned[0].body).points[0] : null;
  check("a click on a turned photo is posted as a pixel of the photo",
    turnedPin !== null && turnedPin.name === "Vega" &&
      Math.abs(turnedPin.x - 700) < 1e-6 && Math.abs(turnedPin.y - 1100) < 1e-6,
    turnedPin ? `stage=(${seen.x},${seen.y}) posted=(${turnedPin.x},${turnedPin.y})` : "no pin");

  // Two refits in flight at once.  By the time the slower answer arrives the
  // user has already clicked again, so that answer describes a transform nobody
  // asked for any more; applying it would jump the photo back under the pointer.
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  liveDefer = true;
  forgetPins();                                   // the pin above is Vega's
  liveReply = {
    alignment: { roll_deg: -12.5, mirrored: false, dx: 30, dy: -8 },
    residuals: { Vega: 4 },
    rms_px: 4,
    max_px: 4,
  };
  clickAt(300 * 0.5, 400 * 0.5);                  // pick Vega
  clickAt(500 * 0.5, 300 * 0.5);                  // first pin, answer withheld
  liveReply = {
    alignment: { roll_deg: 6, mirrored: false, dx: 2, dy: 1 },
    residuals: { Vega: 1 },
    rms_px: 1,
    max_px: 1,
  };
  clickAt(300 * 0.5, 400 * 0.5);                  // pick Vega again
  clickAt(520 * 0.5, 310 * 0.5);                  // second pin, answer withheld
  await settle();
  const held = heldLive.slice(-2);
  const deliver = (entry) => entry.resolve(stubResponse(200, "OK", entry.answer));
  if (held.length === 2) deliver(held[1]);
  await settle();
  const afterNewer = page.state.align.roll_deg;
  if (held.length === 2) deliver(held[0]);
  await settle();
  check("a late answer to an older refit cannot move the photo back",
    held.length === 2 && afterNewer === 6 &&
      page.state.align.roll_deg === 6 && page.state.align.dx === 2,
    `held=${held.length} newer=${afterNewer} final=${page.state.align.roll_deg}/${page.state.align.dx}`);
  liveDefer = false;

  // A refusal from the server keeps the server's own wording, with the endpoint
  // and the status in front of it: the status line is the only place the user
  // reads what went wrong, and "request failed" alone would not say which one.
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  forgetPins();                                   // a pinned circle is a removal
  liveReply = null;                               // stub answers 400 + an error
  clickAt(300 * 0.5, 400 * 0.5);                  // pick Vega
  clickAt(500 * 0.5, 300 * 0.5);                  // place it -> refused
  await settle();
  await settle();
  const refused = documentStub.getElementById("status").textContent;
  check("a refused refit shows the endpoint and the server's reason",
    refused === "POST /api/live failed (400): no /api/live reply staged", refused);

  // And the case that actually bit: a server without /api/live answers with an
  // HTML error page.  Read as JSON that is "Unexpected token '<'", which says
  // nothing; it has to come out as "the server is out of date".
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  forgetPins();                                   // a pinned circle is a removal
  liveHtml = true;
  clickAt(300 * 0.5, 400 * 0.5);                  // pick Vega
  clickAt(500 * 0.5, 300 * 0.5);                  // place it -> HTML 404
  await settle();
  await settle();
  const stale = documentStub.getElementById("status").textContent;
  check("an HTML 404 from an older server is reported as a stale server",
    stale.includes("(404)") && stale.includes("restart") &&
      !stale.includes("Unexpected token"),
    stale);
  liveHtml = false;

  console.log(results.join("\n"));
  const failed = results.filter((line) => line.startsWith("FAIL")).length;
  console.log(failed ? `${failed} page checks failed` : "PAGE_CHECK_OK");
  process.exit(failed ? 1 : 0);
})().catch((error) => {
  console.error(String((error && error.stack) || error));
  process.exit(1);
});
