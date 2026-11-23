// JS-Kopie von glassLight() und liquid() aus D:\Dev\nojo-design\nojo-ui.ts (Quelle dort; hier ohne Typen,
// weil die Vergleichsseite ohne Build-Schritt laeuft).
(() => {
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function glassLight() {
    let lit = null;
    document.addEventListener("pointermove", (e) => {
      const g = e.target?.closest?.(".n-glass");
      if (lit && lit !== g) lit.classList.remove("lit");
      lit = g;
      if (!g || reduced()) return;
      const r = g.getBoundingClientRect();
      g.style.setProperty("--lx", `${e.clientX - r.left}px`);
      g.style.setProperty("--ly", `${e.clientY - r.top}px`);
      g.classList.add("lit");
    }, { passive: true });
  }

  const LIGHT_DIR = (() => { const l = Math.hypot(0.45, 0.9); return [-0.45 / l, -0.9 / l]; })();
  const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
  const filters = new Map();
  let defs = null, seq = 0;
  const supported = () => typeof CSS !== "undefined" && CSS.supports("backdrop-filter", "url(#n)") && /Chrome\/\d+/.test(navigator.userAgent);

  function maps(w, h, r, refract, dpr) {
    const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
    const disp = new ImageData(W, H), spec = new ImageData(W, H);
    const hw = w / 2, hh = h / 2, rr = Math.min(r, hw, hh);
    const edgeW = Math.max(1, Math.min(12, Math.min(hw, hh) * 0.65));
    const S = refract * 2 + 2, aa = 0.6 / dpr + 0.25;
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const x = (i + 0.5) / dpr - hw, y = (j + 0.5) / dpr - hh;
      const qx = Math.abs(x) - hw + rr, qy = Math.abs(y) - hh + rr;
      const d = Math.min(Math.max(qx, qy), 0) + Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) - rr;
      let nx, ny;
      if (qx > 0 && qy > 0) { const l = Math.hypot(qx, qy); nx = qx / l; ny = qy / l; }
      else if (qx > qy) { nx = 1; ny = 0; } else { nx = 0; ny = 1; }
      if (x < 0) nx = -nx;
      if (y < 0) ny = -ny;
      const depth = Math.max(-d, 0), edge = 1 - smooth(0, edgeW, depth), off = edge * edge * refract;
      const k = (j * W + i) * 4;
      disp.data[k] = clamp(Math.round(128 - (nx * off / S) * 255), 0, 255);
      disp.data[k + 1] = clamp(Math.round(128 - (ny * off / S) * 255), 0, 255);
      disp.data[k + 2] = 128; disp.data[k + 3] = 255;
      const facing = Math.max(nx * LIGHT_DIR[0] + ny * LIGHT_DIR[1], 0) ** 3;
      const rim = Math.exp(-(((depth - 0.7) / 0.65) ** 2)), cover = 1 - smooth(-aa, aa, d);
      const a = (rim * (0.12 + 0.36 * facing) + 0.03 * edge * facing) * cover;
      spec.data[k] = spec.data[k + 1] = spec.data[k + 2] = 255;
      spec.data[k + 3] = clamp(Math.round(a * 255), 0, 255);
    }
    const url = (img) => { const c = document.createElement("canvas"); c.width = W; c.height = H; c.getContext("2d").putImageData(img, 0, 0); return c.toDataURL("image/png"); };
    return { disp: url(disp), rim: url(spec), scale: S };
  }

  function filter(w, h, r, refract) {
    const dpr = Math.min(3, Math.max(1, devicePixelRatio || 1));
    const key = `${w}x${h}r${r}f${refract}d${dpr}`;
    let f = filters.get(key);
    if (f) { f.refs++; return { key, f }; }
    if (!defs) {
      defs = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      defs.setAttribute("aria-hidden", "true");
      defs.style.cssText = "position:absolute;width:0;height:0;overflow:hidden;pointer-events:none";
      document.body.append(defs);
    }
    const m = maps(w, h, r, refract, dpr), id = `n-lq-${++seq}`;
    defs.insertAdjacentHTML("beforeend",
      `<filter id="${id}" x="0" y="0" width="${w}" height="${h}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">` +
      `<feImage href="${m.disp}" x="0" y="0" width="${w}" height="${h}" preserveAspectRatio="none" result="m"/>` +
      `<feDisplacementMap in="SourceGraphic" in2="m" scale="${m.scale}" xChannelSelector="R" yChannelSelector="G"/></filter>`);
    f = { id, rim: m.rim, refs: 1 };
    filters.set(key, f);
    return { key, f };
  }
  function release(key) {
    const f = key && filters.get(key);
    if (!f || --f.refs > 0) return;
    filters.delete(key);
    defs?.querySelector(`#${f.id}`)?.remove();
  }

  function liquid() {
    if (!supported()) return;
    const keys = new Map(), timers = new WeakMap();
    const drop = (el) => { release(keys.get(el)); keys.delete(el); ro.unobserve(el); };
    const apply = (el) => {
      if (!el.isConnected || !el.classList.contains("n-liquid")) { drop(el); return; }
      const w = Math.round(el.offsetWidth), h = Math.round(el.offsetHeight);
      if (w < 4 || h < 4) return;
      const rt = getComputedStyle(el).borderTopLeftRadius.split(" ")[0];
      const r = Math.round(rt.endsWith("%") ? (parseFloat(rt) / 100) * Math.min(w, h) : parseFloat(rt) || 0);
      const refract = Number(el.dataset.refract) || (el.classList.contains("panel") ? 7 : 4.5);
      const before = keys.get(el);
      const { key, f } = filter(w, h, Math.min(r, Math.floor(Math.min(w, h) / 2)), refract);
      if (before === key) { f.refs--; return; }
      keys.set(el, key);
      el.style.setProperty("--n-lq", `url(#${f.id})`);
      el.style.setProperty("--n-lq-rim", `url("${f.rim}")`);
      el.dataset.lq = "";
      release(before);
    };
    const later = (el, wait = 90) => { clearTimeout(timers.get(el)); timers.set(el, setTimeout(() => apply(el), keys.has(el) ? wait : 0)); };
    const ro = new ResizeObserver((list) => list.forEach((e) => later(e.target)));
    const scan = () => {
      document.querySelectorAll(".n-liquid").forEach((el) => { if (keys.has(el) || timers.has(el)) return; ro.observe(el); later(el, 0); });
      for (const el of [...keys.keys()]) if (!el.isConnected || !el.classList.contains("n-liquid")) drop(el);
    };
    let pending = 0;
    new MutationObserver(() => { cancelAnimationFrame(pending); pending = requestAnimationFrame(scan); })
      .observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class"] });
    scan();
  }

  window.nojo = { glassLight, liquid };
})();
