/* Checks the browser half of the alignment page against the Python half.

   Called by lens_calibration.selfcheck (Py -> node).  Two things are verified
   that Python cannot see from its side:

   1. forward() in the page -- the map that turns raster pixels into overlay
      pixels for the pins the user places -- reproduces
      session.Alignment.forward for a table of transforms computed in Python;
   2. the real event handlers turn a drag, the rotation slider, the mirror box
      and the arrow keys into exactly the alignment the server is sent;
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
      mode or not) or the ✕ in the readout, and the survivors are refitted;
   6. the photograph is shown, not enhanced: no filter until the gamma slider asks
      for one, the one control is the exponent of a single SVG transfer on all
      three channels (a fragment of this page -- nothing fetched, nothing sent to
      the server), flat puts it back, the rings are never dimmed with it, and a
      zoom is measured against the photograph's own pixels rather than the window
      (that is what lets the browser's zoom reach real detail);
   7. the corrected preview: pressing the button asks the server to resample the
      frame onto an ideal equiangular lens (a *render*, so it happens then and not
      on load), the answer is put on the stage at the photograph's own size so the
      comparison is a flash and not a jump, the overlay is asked for through that
      same ideal lens -- on the photograph the rings must mark the photograph's
      own lens -- and nothing can be pinned, dropped or refitted on pixels that
      are no longer the photograph's.

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
    // The SVG transfer's exponent lives here.
    attrs: {},
    setAttribute(name, value) { node.attrs[name] = String(value); },
    getAttribute(name) { return name in node.attrs ? node.attrs[name] : null; },
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
/* The ids the markup actually declares.  A stub that invents a node for any id
   asked of it cannot tell a working getElementById from a broken one -- a lookup
   of an id the page never carries would come back as a node, the handler would
   write to it, and the check would pass while the control did nothing in a real
   browser.  Anything the page looks up has to be in the page. */
const declaredIds = new Set(
  [...html.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
const documentStub = {
  getElementById(id) {
    if (!declaredIds.has(id)) return null;
    return nodes[id] || (nodes[id] = makeNode(id));
  },
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
// /corrected.png answers with a PNG-shaped blob; set correctedReply to stage a
// refusal (the status and JSON body the server would send) instead.
let correctedReply = null;
// URL.createObjectURL is what the page turns the corrected picture into, and the
// only reason the page needs URL at all -- URLSearchParams stays Node's own.
const objectUrls = [];
const urlStub = {
  createObjectURL(blob) {
    objectUrls.push(blob);
    return `blob:corrected/${objectUrls.length}`;
  },
  revokeObjectURL(url) {
    const index = objectUrls.findIndex((_, at) => `blob:corrected/${at + 1}` === url);
    if (index >= 0) objectUrls[index] = null;
  },
};
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
  // An image answer: the page only hands it to URL.createObjectURL, which is
  // stubbed below, so its contents do not matter here -- that the *bytes* come
  // back as a blob and not as JSON is the point.
  blob: async () => ({ type: "image/png", size: 1024 }),
});
const reply = (status, body) => Promise.resolve(stubResponse(status, "", body));
const replyHtml = (status, statusText, html) => Promise.resolve(stubResponse(status, statusText, html));
const fetchStub = (url, init) => {
  calls.push({ url, method: (init && init.method) || "GET", body: init && init.body });
  if (url === "/api/open") {
    return reply(200, { name: "night.jpg", width: 1400, height: 1400 });
  }
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
  if (url.startsWith("/corrected.png")) {
    if (correctedReply) return reply(correctedReply.status, correctedReply.body);
    return reply(200, { image: "corrected" });
  }
  if (sessionReply) return reply(sessionReply.status, sessionReply.body);
  return Promise.reject(new Error("offline"));
};
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

const page = new Function("window", "document", "fetch", "URL",
  source + "\nreturn { state, forward, inverse, draw, applyTransform, ringRadius, openFrame, loadSession,"
    + " sendPins, renderReadout, syncLive, requestPayload, syncProjectionRows,"
    + " resize, applyReveal, applyZoom, setSaveState, setCorrected };" )(
  windowStub, documentStub, fetchStub, urlStub);

/* ---- 1. the raster -> overlay map ---- */
page.state.session = { image: spec.image, overlay: null };
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

/* ---- 2b. the overlay model: the measured lens is the default ---- */
/* The overlay is drawn with the lens the rig measures, so the panel has to be
   able to say what that lens is and the request has to carry it.  The analytic
   families are still there, and their field of view replaces those fields. */
documentStub.getElementById("projection").value = "fisheye";
documentStub.getElementById("focal").value = "582.3";
documentStub.getElementById("kTerms").value = "-0.10455, -0.0008, 0";
page.syncProjectionRows();
const model = page.requestPayload();
check("the measured lens is what the request carries",
  model.projection === "fisheye" && model.focal_px === 582.3 &&
    model.coefficients.length === 3 && Math.abs(model.coefficients[0] + 0.10455) < 1e-9 &&
    documentStub.getElementById("focalRow").hidden === false &&
    documentStub.getElementById("fovRow").hidden === true,
  JSON.stringify(model));
documentStub.getElementById("projection").value = "equidistant";
page.syncProjectionRows();
check("an analytic projection hides the lens fields",
  documentStub.getElementById("focalRow").hidden === true &&
    documentStub.getElementById("kRow").hidden === true &&
    documentStub.getElementById("fovRow").hidden === false,
  `focal=${documentStub.getElementById("focalRow").hidden}`);
documentStub.getElementById("projection").value = "fisheye";
page.syncProjectionRows();

/* ---- 3. the draw path ---- */
const counts = { arc: 0, strokeText: 0, fillText: 0, moveTo: 0, lineTo: 0, fill: 0, moves: [] };
const countingCtx = {
  setTransform() {}, clearRect() {}, save() {}, restore() {}, beginPath() {},
  closePath() {},
  moveTo(x, y) { counts.moveTo += 1; counts.moves.push([x, y]); },
  lineTo() { counts.lineTo += 1; },
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
  // A tilted or off-zenith camera draws a zenith-distance ring as a curve, so
  // the grid arrives as one polyline per ring and per meridian -- and the zenith
  // (where the cross goes) is a separate point from the optical axis.
  zenith: { x: 660, y: 720 },
  grid: {
    rings: [
      { theta_deg: 30, points: [[700, 470], [930, 700], [700, 930]] },
      { theta_deg: 60, points: [[700, 240], [1160, 700], [700, 1160]] },
      { theta_deg: 90, points: [[700, 9], [1391, 700], [700, 1391]] },
    ],
    meridians: [
      { az_deg: 0, label: "N", points: [[700, 9], [700, 350], [700, 700]], label_x: 700, label_y: 8.6 },
      { az_deg: 90, label: "E", points: [[8.6, 700], [350, 700], [700, 700]], label_x: 8.6, label_y: 700 },
      { az_deg: 180, label: "S", points: [[700, 1391.4], [700, 1050], [700, 700]], label_x: 700, label_y: 1391.4 },
      { az_deg: 270, label: "W", points: [[1391.4, 700], [1050, 700], [700, 700]], label_x: 1391.4, label_y: 700 },
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
  counts.moves = [];
  page.draw();
  return {
    arc: counts.arc, strokeText: counts.strokeText, fillText: counts.fillText,
    moveTo: counts.moveTo, lineTo: counts.lineTo, fill: counts.fill,
    moves: counts.moves.slice(),
  };
};

const gridOn = drawAndCount(true, stars);
const gridOff = drawAndCount(false, stars);
const bare = drawAndCount(true, []);
const rings = overlay.grid.rings.length;
const meridians = overlay.grid.meridians.length;

// every grid curve is stroked twice -- a dark halo, then the line -- and each
// pass is one moveTo and one lineTo per segment of the polyline
const curves = overlay.grid.rings.concat(overlay.grid.meridians);
const curveMoves = 2 * curves.length;
const curveLines = 2 * curves.reduce((total, curve) => total + curve.points.length - 1, 0);
check("draw() strokes every grid curve (halo + line)",
  gridOn.moveTo - gridOff.moveTo === curveMoves &&
    gridOn.lineTo - gridOff.lineTo === curveLines,
  `moves=${gridOn.moveTo - gridOff.moveTo}/${curveMoves} lines=${gridOn.lineTo - gridOff.lineTo}/${curveLines}`);
check("draw() crosses the zenith, not the optical axis",
  bare.moves.some(([x, y]) => Math.abs(x - (overlay.zenith.x * 0.5 - 16)) < 0.01 &&
    Math.abs(y - overlay.zenith.y * 0.5) < 0.01),
  JSON.stringify(bare.moves.slice(0, 3)));
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
    image: {
      url: "/frame.jpg", name: "night.jpg", width: 1400, height: 1400,
      frame_width: 2800, frame_height: 2800,
    },
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
    camera: null,
    lens_source: "seed",
    transform_check_px: null,
    counts: { catalogue: 0 },
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
  check("the header reports the file's own size, not just the raster",
    documentStub.getElementById("frameSize").textContent === "2800×2800 · raster 1400",
    documentStub.getElementById("frameSize").textContent);
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
    camera: { roll_deg: 0, centre_px: { x: 700, y: 700 }, mirrored: false },
    lens_source: "seed",
    transform_check_px: 0,
  };
  page.state.ctx = countingCtx;
  page.state.cssW = 700;
  page.state.cssH = 700;
  page.state.scale = 0.5;
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  page.state.live = { on: false, pending: null, points: [], rms: null, max: null };
  page.state.showGrid = false;

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
    Boolean(page.state.live.pending) && page.state.live.pending.name === "Vega",
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
  check("the click posts the star and the pixel of the photograph it was seen at",
    calls.slice(beforePin).filter((call) => call.url === "/api/live").length === 1 &&
      posted.method === "POST" &&
      pinBody.points.length === 1 &&
      pinBody.points[0].name === "Vega" &&
      pinBody.points[0].x === 500 && pinBody.points[0].y === 300 &&
      !("star_x" in pinBody.points[0]),
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
  // A refit that measured no lens leaves nothing to save: the button follows
  // the lens, not the fact that something was refitted.
  check("a refit without a lens leaves nothing to save",
    page.state.live.lens === null && documentStub.getElementById("save").disabled === true);

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
  const lastUndo = calls.slice(beforeLast).filter((call) => call.url === "/api/live");
  check("undoing the last pin tells the server to forget it",
    lastUndo.length === 1 && JSON.parse(lastUndo[0].body).points.length === 0 &&
      page.state.live.points.length === 0,
    `calls=${lastUndo.length} points=${page.state.live.points.length}`);

  /* ---- 5c. taking one pin out, from its circle or from the list ---- */
  /* A pin that turns out to be wrong is far more common than a pin that turns
     out to be the last one, so a single pair has to be droppable on its own:
     clicking the circle that carries it, or the ✕ beside it in the readout.
     Both refit through what is left -- the server never nudges an existing fit,
     so the removed pair leaves nothing behind -- and with nothing left nothing
     is posted at all, which is the page's own "forget them". */
  const pinSet = [
    { name: "Vega", x: 500, y: 300, residual_px: 2 },
    { name: "Altair", x: 890, y: 430, residual_px: 1.4 },
    { name: "Deneb", x: 610, y: 990, residual_px: 1.1 },
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
  const emptyDrop = calls.slice(beforeEmpty).filter((call) => call.url === "/api/live");
  check("removing the last pin forgets the pins and clears the lens",
    emptyDrop.length === 1 && JSON.parse(emptyDrop[0].body).points.length === 0 &&
      page.state.live.points.length === 0 && page.state.live.lens === null &&
      !page.state.live.pending,
    `calls=${emptyDrop.length} points=${page.state.live.points.length} lens=${page.state.live.lens}`);

  /* ---- 5b. the fourth pin hands over to the lens ---- */
  /* Four pins are the point where the server stops moving the photo and starts
     solving the fisheye: it answers with the centre, focal length and k terms,
     plus the overlay re-projected through them.  The page has to draw that
     overlay (rings on the calibrated positions) while still remembering each
     ring's *ideal* position, because that is what a pin's field angle is read
     from -- posting the drawn position instead would make the fifth pin feed the
     fit its own output. */
  const threePins = [
    { name: "Vega", x: 500, y: 300, residual_px: 2 },
    { name: "Altair", x: 890, y: 430, residual_px: 2 },
    { name: "Deneb", x: 610, y: 990, residual_px: 2 },
  ];
  const fourPins = threePins.concat([
    { name: "Polaris", x: 990, y: 1040, residual_px: 1.1 },
  ]);
  page.state.live.points = threePins;
  page.state.live.on = true;
  page.syncLive();
  check("three pins say one more star measures the lens",
    liveNote().includes("one more star solves the lens"), liveNote());

  // The calibrated overlay is the same rings where the *camera* puts them: the
  // server has already done the projection, so x/y is where a ring is drawn.
  const calibrated = liveStars.map((star) => ({ ...star, x: star.x + 6, y: star.y - 4 }));
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

  // A pin taken against the calibrated overlay posts the same two things as any
  // other: the star's name and the pixel of the photograph.  Neither of them is a
  // position the overlay produced, which is what stops the fit feeding on its own
  // output.
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
  // The click lands on a stage pixel and the pin is the photograph pixel that
  // lands there -- under the transform the photo has *now*, which the previous
  // refit set.  Nothing about the ring's drawn position goes on the wire.
  const wantFifth = page.inverse(250 / 0.5, 150 / 0.5);
  check("a pin on the calibrated overlay posts the name and the photo pixel",
    Boolean(picked) && picked.name === "Vega" && !("star_x" in picked) &&
      fifthBody.points.length === 4 &&
      fifthBody.points.some((point) => point.name === "Vega" &&
        !("star_x" in point) &&
        Math.abs(point.x - wantFifth.x) < 1e-9 && Math.abs(point.y - wantFifth.y) < 1e-9),
    `picked=${JSON.stringify(picked)} posted=${JSON.stringify(fifthBody.points)}`);

  // When the fit is capped -- every odd term the series carries is already in
  // play -- another star is another equation for the same unknowns, and the
  // readout repeats the server's note about it rather than promising a
  // coefficient that will never appear.
  const sixPins = fourPins.concat([
    { name: "Sirius", x: 615, y: 305, residual_px: 0.8 },
  ]);
  const capped = {
    pins: 8, num_coefficients: 4, focal_px: 512.3,
    coefficients: [-0.0538, 0.0021, -0.00031, 0.00002],
    centre_px: { x: 678.5, y: 707.9 }, covers_sigma: 0.85,
    lens: { theta_sigma_edge_deg: 88.62 },
    notes: ["8 pins, but the fit stops at the theta^9 term: stars past the first 7"
      + " over-determine the same unknowns rather than adding coefficients"],
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
  check("a star past the cap adds equations, not a coefficient",
    Boolean(page.state.live.lens) && page.state.live.lens.num_coefficients === 4 &&
      liveNote().includes("4 k terms") &&
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

  /* ---- 6. the photograph is shown, not enhanced ---- */
  // The server hands over the file itself, full size and with no tone curve: a
  // night frame's lift is a *view*, so it is one plain CSS filter on the picture,
  // flat until asked for, and the rings -- which are the things being measured
  // against -- must never pass through it.  Nothing about it is sent anywhere.
  page.state.session = {
    image: { width: 1400, height: 1400, frame_width: 4000, frame_height: 4000 },
    overlay: null,
  };
  const frameNode = documentStub.getElementById("frame");
  const channel = (id) => documentStub.getElementById(id);
  page.state.reveal = { gamma: 1 };
  page.applyReveal();
  check("the photograph starts flat: no filter at all",
    !frameNode.style.filter, `filter=${frameNode.style.filter}`);
  // The transfer only works if the page can find its channels and if it transfers
  // sRGB: getElementById on an id the markup does not carry returns null, and a
  // transfer over linear light would turn gamma 0.45 into a blown-out frame.
  check("the gamma transfer's channels carry the ids the page looks up",
    ["feFuncR", "feFuncG", "feFuncB"].every((id) =>
      new RegExp(`<${id} id="${id}" type="gamma"`).test(html)),
    ["feFuncR", "feFuncG", "feFuncB"].join(" "));
  check("the transfer is over sRGB, not linear light",
    /<filter id="reveal" color-interpolation-filters="sRGB">/.test(html), "");

  const wireBefore = calls.length;
  documentStub.getElementById("gamma").fire("input", { target: { value: "0.45" } });
  check("the slider is the exponent of one transfer, on all three channels",
    frameNode.style.filter === "url(#reveal)" &&
      channel("feFuncR").attrs.exponent === "0.45" &&
      channel("feFuncG").attrs.exponent === "0.45" &&
      channel("feFuncB").attrs.exponent === "0.45",
    `filter=${frameNode.style.filter} g=${JSON.stringify(channel("feFuncG").attrs)}`);
  check("the filter is a fragment of this page, nothing to fetch",
    frameNode.style.filter === "url(#reveal)" && html.includes('id="reveal"'),
    `filter=${frameNode.style.filter}`);
  check("the lift never reaches the server", calls.length === wireBefore,
    `requests=${calls.length - wireBefore}`);
  check("the rings are not dimmed with the photograph",
    !documentStub.getElementById("overlay").style.filter,
    `overlay filter=${documentStub.getElementById("overlay").style.filter}`);
  check("the slider reports what it is doing",
    documentStub.getElementById("gammaOut").textContent === "0.45",
    documentStub.getElementById("gammaOut").textContent);

  documentStub.getElementById("flat").fire("click", {});
  check("flat goes back to the file as written, slider included",
    page.state.reveal.gamma === 1 && !frameNode.style.filter &&
      documentStub.getElementById("gamma").value === "1" &&
      channel("feFuncG").attrs.exponent === "1",
    `reveal=${JSON.stringify(page.state.reveal)} filter=${frameNode.style.filter}`);

  // Zoom is measured against the photograph, not against the window: 100 % is one
  // css pixel per pixel of the file, which is the only size at which the
  // browser's own zoom has real detail to magnify.  Fit takes back whatever the
  // browser zoom hands out, so leaning in there would enlarge a downscaled copy.
  documentStub.getElementById("zoom").fire("change", { target: { value: "1" } });
  check("100 % sizes the stage by the photograph's own pixels",
    page.state.cssW === 4000 && page.state.cssH === 4000 &&
      Math.abs(page.state.scale - 4000 / 1400) < 1e-9,
    `css=${page.state.cssW}x${page.state.cssH} scale=${page.state.scale}`);
  documentStub.getElementById("zoom").fire("change", { target: { value: "2" } });
  const overlayNode = documentStub.getElementById("overlay");
  check("a zoomed stage cannot ask for an unreasonable canvas",
    page.state.cssW === 8000 && overlayNode.width <= 4096 && overlayNode.height <= 4096,
    `css=${page.state.cssW} canvas=${overlayNode.width}x${overlayNode.height}`);
  documentStub.getElementById("zoom").fire("change", { target: { value: "fit" } });
  check("fit hands the stage back to the window",
    page.state.zoom === null, `zoom=${page.state.zoom}`);

  /* ---- 7. the corrected preview ---- */
  /* The stage can also show the frame resampled onto an ideal equiangular lens.
     Four statements: the picture is *rendered* by the server when the button is
     pressed (not linked, not on page load), it arrives at the photograph's own
     size so the swap is a flash comparison rather than a jump, the overlay is
     asked for through the same ideal lens -- on the photograph the rings have to
     mark the photograph's own lens -- and nothing can be pinned on pixels that
     are no longer the photograph's. */
  page.state.align = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  page.state.overlayAlign = { roll_deg: 0, mirrored: false, dx: 0, dy: 0 };
  // The way back rebuilds the full session query, so the field of view has to
  // read like a filled-in field rather than like a stub's ``undefined``.
  documentStub.getElementById("fov").value = "180";
  page.state.live = {
    ...page.state.live, on: false, pending: null, points: [], overlay: null, lens: null,
  };
  const withOverlay = {
    ...sessionPayload,
    image: { url: "/frame.jpg", name: "night.jpg", width: 1400, height: 1400,
             frame_width: 2800, frame_height: 2800 },
    overlay: {
      calibrated: true, count: 1, mag_limit: 4.5, projection: "fisheye",
      radius_px: 690, theta_edge_deg: 98.0, fov_deg: 195.9,
      centre: { x: 700, y: 700 }, zenith: { x: 700, y: 700 },
      grid: { rings: [], meridians: [] },
      stars: [{ name: "Vega", x: 700, y: 700, vmag: 0.5, alt_deg: 80, theta_deg: 10 }],
      aids: null,
    },
    overlay_error: null,
    lens_source: "pins",
    corrected: { url: "/corrected.png", active: false, projection: "equiangular" },
  };
  const correctedButton = documentStub.getElementById("corrected");
  page.state.session = { ...withOverlay, lens_source: "seed" };
  page.setSaveState();
  check("the corrected button waits for a measured lens",
    correctedButton.disabled === true, `disabled=${correctedButton.disabled}`);

  page.state.session = withOverlay;
  sessionReply = { status: 200, body: withOverlay };
  page.setSaveState();
  check("a measured lens is what the correction needs, so it enables the button",
    correctedButton.disabled === false &&
      correctedButton.textContent === "view corrected image",
    `disabled=${correctedButton.disabled} label=${correctedButton.textContent}`);

  // A stage of a known size: the swap must not change it, and the hit test below
  // is css pixels against raster pixels.
  page.state.zoom = 1;
  page.applyZoom();
  const stageBefore = `${page.state.cssW}x${page.state.cssH}`;
  const wireBeforeCorrected = calls.length;
  correctedButton.fire("click", {});
  await settle();
  await settle();
  const correctedCalls = calls.slice(wireBeforeCorrected);
  const renderCall = correctedCalls.find((call) => call.url.startsWith("/corrected.png"));
  const renderQuery = new URLSearchParams(renderCall ? renderCall.url.split("?")[1] : "");
  check("pressing the button asks the server to render the corrected frame",
    Boolean(renderCall) && renderQuery.get("projection") === "fisheye",
    renderCall ? renderCall.url : "no /corrected.png call");
  check("the corrected picture comes back as an image, not as JSON",
    objectUrls.length === 1 && objectUrls[0] !== null && frameNode.src === "blob:corrected/1",
    `src=${frameNode.src}`);
  const overlayCall = correctedCalls[correctedCalls.length - 1];
  check("the overlay is redrawn through the same ideal lens",
    Boolean(overlayCall) && overlayCall.url.startsWith("/api/session") &&
      new URLSearchParams(overlayCall.url.split("?")[1]).get("corrected") === "1",
    overlayCall ? overlayCall.url : "no /api/session call");
  check("the swap is a flash comparison: the stage keeps its size and place",
    `${page.state.cssW}x${page.state.cssH}` === stageBefore &&
      page.state.align.dx === 0 && page.state.align.dy === 0,
    `${stageBefore} -> ${page.state.cssW}x${page.state.cssH}`);
  check("the readout says which picture the rings belong to",
    documentStub.getElementById("readout").innerHTML.includes("ideal equiangular"),
    documentStub.getElementById("readout").innerHTML.slice(0, 60));
  check("the button offers the way back",
    correctedButton.textContent === "back to the photo",
    `label=${correctedButton.textContent}`);

  // Pinning is a pixel of the photograph.  The circle is right under the pointer
  // -- the same click picks it once the photo is back -- so what stops it here is
  // the view, not a stray click.
  const hit = { clientX: 100, clientY: 100 };
  const wireBeforeClick = calls.length;
  stageNode.fire("pointerdown", { button: 0, pointerId: 77, ...hit });
  stageNode.fire("pointerup", { pointerId: 77 });
  documentStub.getElementById("live").fire("click", {});
  check("nothing is picked or refitted on pixels of the ideal lens",
    page.state.live.pending === null && page.state.live.on === false &&
      calls.length === wireBeforeClick,
    `pending=${JSON.stringify(page.state.live.pending)} on=${page.state.live.on}`);

  correctedButton.fire("click", {});
  await settle();
  await settle();
  const backCall = calls.slice(-2).reverse().find((call) => call.url.startsWith("/api/session"));
  check("going back puts the photograph and the measured overlay up again",
    page.state.corrected === false && frameNode.src.startsWith("/frame.jpg") &&
      Boolean(backCall) &&
      new URLSearchParams(backCall.url.split("?")[1]).get("corrected") === "0",
    `src=${frameNode.src} session=${backCall ? backCall.url : "none"}`);
  check("the corrected picture's object URL is released",
    objectUrls[0] === null, JSON.stringify(objectUrls));

  page.state.live.on = true;
  const raster = 100 / page.state.scale;
  page.state.session.overlay.stars[0].x = raster;
  page.state.session.overlay.stars[0].y = raster;
  page.state.overlayAlign = { ...page.state.align };
  stageNode.fire("pointerdown", { button: 0, pointerId: 78, ...hit });
  stageNode.fire("pointerup", { pointerId: 78 });
  check("the same click does pick the star once the photograph is back",
    Boolean(page.state.live.pending) && page.state.live.pending.name === "Vega",
    `pending=${JSON.stringify(page.state.live.pending)}`);
  page.state.live.on = false;
  page.state.live.pending = null;

  // A server that cannot render it must say so, not leave a broken picture on
  // the stage under rings drawn for a correction that is not there.
  correctedReply = { status: 400, body: { error: "cannot read the frame" } };
  correctedButton.fire("click", {});
  await settle();
  await settle();
  check("a refusal is reported and the photograph stays up",
    page.state.corrected === false && frameNode.src.startsWith("/frame.jpg") &&
      documentStub.getElementById("status").textContent.includes("cannot read the frame"),
    `src=${frameNode.src} status=${documentStub.getElementById("status").textContent}`);
  correctedReply = null;

  console.log(results.join("\n"));
  const failed = results.filter((line) => line.startsWith("FAIL")).length;
  console.log(failed ? `${failed} page checks failed` : "PAGE_CHECK_OK");
  process.exit(failed ? 1 : 0);
})().catch((error) => {
  console.error(String((error && error.stack) || error));
  process.exit(1);
});
