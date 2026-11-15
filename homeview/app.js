(() => {
  "use strict";

  // Safe DOM builder — never injects unescaped HTML
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "className") node.className = v;
        else if (k === "text") node.textContent = String(v);
        else if (k.startsWith("on") && typeof v === "function") {
          node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (k === "dataset" && typeof v === "object") {
          for (const [dk, dv] of Object.entries(v)) node.dataset[dk] = String(dv);
        } else {
          node.setAttribute(k, String(v));
        }
      }
    }
    if (Array.isArray(children)) {
      for (const child of children) {
        if (child === null || child === undefined || child === false) continue;
        node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
      }
    } else if (typeof children === "string") {
      node.textContent = children;
    }
    return node;
  }

  function hexToRgb(hex) {
    const clean = (hex || "#e6e4dd").replace("#", "").trim();
    const num = parseInt(clean.length === 6 ? clean : "e6e4dd", 16);
    return [
      ((num >> 16) & 255) / 255,
      ((num >> 8) & 255) / 255,
      (num & 255) / 255,
    ];
  }

  // 4x4 Matrix Math for WebGL
  function mat4Identity() {
    return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
  }
  function mat4Perspective(fovRad, aspect, near, far) {
    const f = 1.0 / Math.tan(fovRad / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }
  function mat4LookAt(eye, target, up) {
    let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
    let len = Math.hypot(zx, zy, zz) || 1;
    zx /= len; zy /= len; zz /= len;
    let xx = up[1] * zz - up[2] * zy;
    let xy = up[2] * zx - up[0] * zz;
    let xz = up[0] * zy - up[1] * zx;
    len = Math.hypot(xx, xy, xz) || 1;
    xx /= len; xy /= len; xz /= len;
    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;
    return new Float32Array([
      xx, yx, zx, 0,
      xy, yy, zy, 0,
      xz, yz, zz, 0,
      -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
      -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
      -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
      1,
    ]);
  }
  function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = a[i], ai1 = a[i + 4], ai2 = a[i + 8], ai3 = a[i + 12];
      out[i]      = ai0 * b[0]  + ai1 * b[1]  + ai2 * b[2]  + ai3 * b[3];
      out[i + 4]  = ai0 * b[4]  + ai1 * b[5]  + ai2 * b[6]  + ai3 * b[7];
      out[i + 8]  = ai0 * b[8]  + ai1 * b[9]  + ai2 * b[10] + ai3 * b[11];
      out[i + 12] = ai0 * b[12] + ai1 * b[13] + ai2 * b[14] + ai3 * b[15];
    }
    return out;
  }

  // =========================================================================
  //  WebGL 3D Apartment Renderer (Sweet Home 3D Scene + Live HA Lighting)
  // =========================================================================
  class Apartment3DEngine {
    constructor(canvas, onPinProjected, onRoomProjected) {
      this.canvas = canvas;
      this.gl = canvas.getContext("webgl", { antialias: true, alpha: false, depth: true });
      this.onPinProjected = onPinProjected;
      this.onRoomProjected = onRoomProjected;
      this.scene = null;
      this.opaqueBatches = [];
      this.glassBatches = [];
      this.wallDefs = [];
      this.wallCuts = {};
      this.lightTargets = {};
      this.lightCurrent = {};
      this.lightAnchors = [];

      this.cutawayEnabled = true;
      this.autoOrbit = false;
      this.mode = "orbit";
      this.target = [627, 42, -597];
      this.distance = 1160;
      this.azimuth = -0.52;
      this.polar = 0.86;
      this.fov = 44;
      this.eye = [627, 860, 120];
      this.vpMatrix = mat4Identity();

      this.flight = null;
      this.pointers = new Map();
      this.lastPinchDist = 0;
      this.lastDrag = null;

      if (this.gl) {
        this._initShaders();
        this._bindInteraction();
      }
    }

    _initShaders() {
      const gl = this.gl;
      const vsSource = `
        attribute vec3 aPos;
        attribute vec3 aNormal;
        attribute vec3 aColor;
        attribute vec3 aMaterial; // x: roughness, y: metalness, z: woodFlag/emissive
        uniform mat4 uVP;
        uniform vec3 uScaleY; // x: baseZ, y: yScale, z: 0
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec3 vColor;
        varying vec3 vMat;
        void main() {
          vec3 wp = aPos;
          if (uScaleY.y > 0.0) {
            wp.y = min(wp.y, uScaleY.x);
          }
          vWorldPos = wp;
          vNormal = normalize(aNormal);
          vColor = aColor;
          vMat = aMaterial;
          gl_Position = uVP * vec4(wp, 1.0);
        }
      `;
      const fsSource = `
        precision highp float;
        varying vec3 vWorldPos;
        varying vec3 vNormal;
        varying vec3 vColor;
        varying vec3 vMat;
        uniform vec3 uEye;
        uniform float uOpacity;
        uniform vec3 uLightPos[6];
        uniform vec3 uLightCol[6];
        uniform float uLightInt[6];

        void main() {
          vec3 N = normalize(vNormal);
          vec3 V = normalize(uEye - vWorldPos);
          vec3 baseCol = vColor;

          // Subtle natural oak plank variation on wood floors
          if (vMat.z > 1.5 && abs(N.y) > 0.8) {
            float plankX = floor(vWorldPos.x / 16.5);
            float plankZ = floor((vWorldPos.z + plankX * 37.0) / 135.0);
            float h = fract(sin(plankX * 12.9898 + plankZ * 78.233) * 43758.5453);
            float seamX = smoothstep(0.0, 0.06, fract(vWorldPos.x / 16.5));
            float grain = sin(vWorldPos.z * 0.18 + h * 6.28) * 0.018;
            baseCol = baseCol * (0.94 + h * 0.09 + grain) * (0.93 + 0.07 * seamX);
          }

          // Architectural hemisphere + key/fill directional lighting
          vec3 sunDir = normalize(vec3(0.42, 0.82, -0.38));
          vec3 fillDir = normalize(vec3(-0.55, 0.55, 0.62));
          float ndl = max(dot(N, sunDir), 0.0);
          float ndf = max(dot(N, fillDir), 0.0);
          float hemi = 0.5 + 0.5 * N.y;
          vec3 ambient = mix(vec3(0.24, 0.26, 0.31), vec3(0.46, 0.47, 0.50), hemi);
          vec3 lit = baseCol * (ambient + ndl * vec3(0.56, 0.55, 0.52) + ndf * vec3(0.20, 0.23, 0.28));

          // Specular highlight for metallic/glass/screen surfaces
          vec3 H = normalize(sunDir + V);
          float specPow = mix(12.0, 64.0, 1.0 - vMat.x);
          float spec = pow(max(dot(N, H), 0.0), specPow) * (0.08 + vMat.y * 0.35);
          lit += vec3(spec);

          // 6 Real-Time Home Assistant Room Point Lights
          for (int i = 0; i < 6; i++) {
            float intensity = uLightInt[i];
            if (intensity > 0.01) {
              vec3 lVec = uLightPos[i] - vWorldPos;
              float dist = length(lVec);
              vec3 L = lVec / max(dist, 1.0);
              float atten = 1.0 / (1.0 + 0.0014 * dist + 0.000018 * dist * dist);
              float diff = max(dot(N, L), 0.0) * 0.72 + 0.28;
              lit += baseCol * uLightCol[i] * (intensity * atten * diff * 1.45);
              // Warm specular bloom on floor & nearby surfaces
              vec3 Hl = normalize(L + V);
              float pSpec = pow(max(dot(N, Hl), 0.0), 22.0) * atten * intensity * 0.25;
              lit += uLightCol[i] * pSpec;
            }
          }

          // Emissive lamp shades / LED strips
          if (vMat.z > 0.1 && vMat.z < 1.4) {
            lit = mix(lit, vec3(1.0, 0.89, 0.62), 0.78);
          }

          // Subtle tone mapping
          lit = lit / (vec3(1.0) + lit * 0.16);
          gl_FragColor = vec4(lit, uOpacity);
        }
      `;

      const compile = (type, src) => {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        return s;
      };
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
      gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
      gl.linkProgram(prog);
      gl.useProgram(prog);
      this.prog = prog;

      this.locs = {
        aPos: gl.getAttribLocation(prog, "aPos"),
        aNormal: gl.getAttribLocation(prog, "aNormal"),
        aColor: gl.getAttribLocation(prog, "aColor"),
        aMaterial: gl.getAttribLocation(prog, "aMaterial"),
        uVP: gl.getUniformLocation(prog, "uVP"),
        uScaleY: gl.getUniformLocation(prog, "uScaleY"),
        uEye: gl.getUniformLocation(prog, "uEye"),
        uOpacity: gl.getUniformLocation(prog, "uOpacity"),
        uLightPos: gl.getUniformLocation(prog, "uLightPos[0]"),
        uLightCol: gl.getUniformLocation(prog, "uLightCol[0]"),
        uLightInt: gl.getUniformLocation(prog, "uLightInt[0]"),
      };
    }

    loadScene(sceneData) {
      this.scene = sceneData;
      this.lightAnchors = (sceneData.smartDevices || []).filter(d => d.category === "light").slice(0, 6);
      for (const l of this.lightAnchors) {
        this.lightTargets[l.key] = 0.0;
        this.lightCurrent[l.key] = 0.0;
      }
      this._buildBuffers();
      if (sceneData.cameras && sceneData.cameras.length) {
        this.setCamera(sceneData.cameras[0].id, false);
      }
    }

    updateLightStates(lightsArray) {
      if (!Array.isArray(lightsArray)) return;
      for (const l of lightsArray) {
        this.lightTargets[l.id] = l.is_on ? 1.0 : 0.0;
      }
    }

    _createBatch(verts, indices, opacity) {
      const gl = this.gl;
      if (!gl || !indices.length) return null;
      const vbo = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);
      const ibo = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
      return { vbo, ibo, count: indices.length, opacity: opacity ?? 1.0 };
    }

    _buildBuffers() {
      const gl = this.gl;
      if (!gl || !this.scene) return;
      this.opaqueBatches = [];
      this.glassBatches = [];
      this.wallDefs = [];

      // 1. Room Floors + Base Slab
      const floorVerts = [];
      const floorIndices = [];
      const pushVert = (arr, px, py, pz, nx, ny, nz, r, g, b, rough, metal, flag) => {
        const idx = arr.length / 12;
        arr.push(px, py, pz, nx, ny, nz, r, g, b, rough, metal, flag);
        return idx;
      };

      for (const rm of this.scene.rooms) {
        const [r, g, b] = hexToRgb(rm.color);
        const woodFlag = rm.wood ? 2.0 : 0.0;
        const baseIdx = floorVerts.length / 12;
        for (const [px, py] of rm.points) {
          pushVert(floorVerts, px, 0.2, py, 0, 1, 0, r, g, b, 0.55, 0.04, woodFlag);
        }
        for (let i = 0; i < rm.tris.length; i += 3) {
          // Ensure upward normal winding in 3D (X, Y_up, Z_plan)
          const i0 = rm.tris[i], i1 = rm.tris[i + 1], i2 = rm.tris[i + 2];
          const p0 = rm.points[i0], p1 = rm.points[i1], p2 = rm.points[i2];
          const crossY = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
          if (crossY < 0) {
            floorIndices.push(baseIdx + i0, baseIdx + i1, baseIdx + i2);
          } else {
            floorIndices.push(baseIdx + i0, baseIdx + i2, baseIdx + i1);
          }
        }
      }
      const floorBatch = this._createBatch(floorVerts, floorIndices, 1.0);
      if (floorBatch) this.opaqueBatches.push(floorBatch);

      // 2. Walls (individual batches so cutaway height can animate smoothly per wall!)
      for (const w of this.scene.walls) {
        const [x1, z1] = w.start;
        const [x2, z2] = w.end;
        const dx = x2 - x1, dz = z2 - z1;
        const len = Math.hypot(dx, dz) || 1;
        const ux = dx / len, uz = dz / len;
        const nx = -uz, nz = ux;
        const halfT = (w.thickness || 15) * 0.5;
        const [sr, sg, sb] = hexToRgb(w.color);
        const [tr, tg, tb] = hexToRgb(w.topColor);
        const wVerts = [];
        const wIdx = [];

        const addBox = (u0, u1, y0, y1) => {
          const ax = x1 + ux * u0, az = z1 + uz * u0;
          const bx = x1 + ux * u1, bz = z1 + uz * u1;
          const c = [
            [ax + nx * halfT, y0, az + nz * halfT],
            [bx + nx * halfT, y0, bz + nz * halfT],
            [bx + nx * halfT, y1, bz + nz * halfT],
            [ax + nx * halfT, y1, az + nz * halfT],
            [ax - nx * halfT, y0, az - nz * halfT],
            [bx - nx * halfT, y0, bz - nz * halfT],
            [bx - nx * halfT, y1, bz - nz * halfT],
            [ax - nx * halfT, y1, az - nz * halfT],
          ];
          const quad = (p0, p1, p2, p3, qnx, qny, qnz, cr, cg, cb) => {
            const bi = wVerts.length / 12;
            pushVert(wVerts, p0[0], p0[1], p0[2], qnx, qny, qnz, cr, cg, cb, 0.72, 0.02, 0);
            pushVert(wVerts, p1[0], p1[1], p1[2], qnx, qny, qnz, cr, cg, cb, 0.72, 0.02, 0);
            pushVert(wVerts, p2[0], p2[1], p2[2], qnx, qny, qnz, cr, cg, cb, 0.72, 0.02, 0);
            pushVert(wVerts, p3[0], p3[1], p3[2], qnx, qny, qnz, cr, cg, cb, 0.72, 0.02, 0);
            wIdx.push(bi, bi + 1, bi + 2, bi, bi + 2, bi + 3);
          };
          quad(c[0], c[1], c[2], c[3], nx, 0, nz, sr, sg, sb);
          quad(c[5], c[4], c[7], c[6], -nx, 0, -nz, sr, sg, sb);
          quad(c[3], c[2], c[6], c[7], 0, 1, 0, tr, tg, tb);
          quad(c[4], c[0], c[3], c[7], -ux, 0, -uz, sr, sg, sb);
          quad(c[1], c[5], c[6], c[2], ux, 0, uz, sr, sg, sb);
        };

        for (const [u0, u1, y0, y1] of w.spans) {
          addBox(u0, u1, y0, y1);
        }
        const batch = this._createBatch(wVerts, wIdx, 1.0);
        if (batch) {
          this.wallDefs.push({
            id: w.id,
            batch,
            midX: (x1 + x2) * 0.5,
            midZ: (z1 + z2) * 0.5,
            height: w.height || 243.84,
            exterior: w.exterior,
          });
          this.wallCuts[w.id] = w.height || 243.84;
        }
      }

      // 3. Furniture & Door/Window 3D Models (grouped into static batches)
      let opVerts = [];
      let opIndices = [];
      const flushOpaque = () => {
        if (opIndices.length) {
          const b = this._createBatch(opVerts, opIndices, 1.0);
          if (b) this.opaqueBatches.push(b);
          opVerts = [];
          opIndices = [];
        }
      };

      for (const item of this.scene.items) {
        // When cutaway is active, skip tiny ceiling-mounted fixtures so they don't float in midair
        const modelObj = this.scene.models[item.model];
        if (!modelObj || !modelObj.submeshes) continue;
        const scaleDiv = modelObj.scale || 1000;
        const cosA = Math.cos(item.angle || 0);
        const sinA = Math.sin(item.angle || 0);

        for (const sub of modelObj.submeshes) {
          let hex = sub.color;
          if (item.materials && item.materials[sub.mtl]) {
            hex = item.materials[sub.mtl];
          } else if (item.color && (sub.frame || (item.kind === "furniture" && sub.tintable))) {
            hex = item.color;
          }
          const [cr, cg, cb] = hexToRgb(hex);
          const isGlass = sub.glass || (sub.opacity && sub.opacity < 0.92);
          const pos = sub.v;
          const idx = sub.i;
          const flag = sub.emissive > 0.2 ? 1.0 : 0.0;

          const targetVerts = isGlass ? [] : opVerts;
          const targetIdx = isGlass ? [] : opIndices;
          if (!isGlass && targetVerts.length / 12 + idx.length > 60000) {
            flushOpaque();
          }

          // Transform model vertices to world space
          const worldPts = new Array(pos.length / 3);
          for (let v = 0; v < pos.length; v += 3) {
            const lx = (pos[v] / scaleDiv) * item.w;
            const ly = (pos[v + 1] / scaleDiv) * item.h;
            const lz = (pos[v + 2] / scaleDiv) * item.d;
            const wx = item.x + lx * cosA - lz * sinA;
            const wy = item.z + ly;
            const wz = item.y + lx * sinA + lz * cosA;
            worldPts[v / 3] = [wx, wy, wz];
          }

          for (let t = 0; t < idx.length; t += 3) {
            const p0 = worldPts[idx[t]];
            const p1 = worldPts[idx[t + 1]];
            const p2 = worldPts[idx[t + 2]];
            if (!p0 || !p1 || !p2) continue;
            const ux = p1[0] - p0[0], uy = p1[1] - p0[1], uz = p1[2] - p0[2];
            const vx = p2[0] - p0[0], vy = p2[1] - p0[1], vz = p2[2] - p0[2];
            let nx = uy * vz - uz * vy;
            let ny = uz * vx - ux * vz;
            let nz = ux * vy - uy * vx;
            const nl = Math.hypot(nx, ny, nz) || 1;
            nx /= nl; ny /= nl; nz /= nl;
            const bi = targetVerts.length / 12;
            pushVert(targetVerts, p0[0], p0[1], p0[2], nx, ny, nz, cr, cg, cb, sub.roughness, sub.metalness, flag);
            pushVert(targetVerts, p1[0], p1[1], p1[2], nx, ny, nz, cr, cg, cb, sub.roughness, sub.metalness, flag);
            pushVert(targetVerts, p2[0], p2[1], p2[2], nx, ny, nz, cr, cg, cb, sub.roughness, sub.metalness, flag);
            targetIdx.push(bi, bi + 1, bi + 2);
          }

          if (isGlass && targetIdx.length) {
            const gb = this._createBatch(targetVerts, targetIdx, sub.opacity || 0.22);
            if (gb) this.glassBatches.push(gb);
          }
        }
      }
      flushOpaque();
    }

    setCamera(camId, animate = true) {
      if (!this.scene) return;
      const cam = (this.scene.cameras || []).find(c => c.id === camId) || this.scene.cameras[0];
      if (!cam) return;

      let destEye, destTarget, destFov, destMode, destCutaway;
      if (cam.mode === "orbit") {
        destMode = "orbit";
        destTarget = [...cam.target];
        destFov = cam.fov || 44;
        destCutaway = Boolean(cam.cutaway);
        const r = cam.distance;
        const pol = cam.polar;
        const az = cam.azimuth;
        destEye = [
          destTarget[0] + r * Math.sin(pol) * Math.sin(az),
          destTarget[1] + r * Math.cos(pol),
          destTarget[2] + r * Math.sin(pol) * Math.cos(az),
        ];
        this.distance = r;
        this.polar = pol;
        this.azimuth = az;
      } else {
        destMode = "interior";
        destEye = [...cam.eye];
        destTarget = [...cam.target];
        destFov = cam.fov || 60;
        destCutaway = false;
      }

      this.cutawayEnabled = destCutaway;
      if (!animate) {
        this.mode = destMode;
        this.eye = destEye;
        this.target = destTarget;
        this.fov = destFov;
        this.flight = null;
        return;
      }

      this.flight = {
        startTime: performance.now(),
        duration: 780,
        fromEye: [...this.eye],
        fromTarget: [...this.target],
        fromFov: this.fov,
        toEye: destEye,
        toTarget: destTarget,
        toFov: destFov,
        toMode: destMode,
      };
    }

    focusPoint(x, y, z) {
      const destTarget = [x, z + 45, y];
      const r = 460;
      const pol = 0.88;
      const az = this.azimuth;
      const destEye = [
        destTarget[0] + r * Math.sin(pol) * Math.sin(az),
        destTarget[1] + r * Math.cos(pol),
        destTarget[2] + r * Math.sin(pol) * Math.cos(az),
      ];
      this.distance = r;
      this.polar = pol;
      this.flight = {
        startTime: performance.now(),
        duration: 650,
        fromEye: [...this.eye],
        fromTarget: [...this.target],
        fromFov: this.fov,
        toEye: destEye,
        toTarget: destTarget,
        toFov: 44,
        toMode: "orbit",
      };
    }

    zoomBy(factor) {
      this.autoOrbit = false;
      if (this.mode === "orbit") {
        this.distance = Math.max(240, Math.min(2100, this.distance * factor));
        this._syncOrbitEye();
      } else {
        this.fov = Math.max(32, Math.min(78, this.fov * factor));
      }
    }

    _syncOrbitEye() {
      this.eye = [
        this.target[0] + this.distance * Math.sin(this.polar) * Math.sin(this.azimuth),
        this.target[1] + this.distance * Math.cos(this.polar),
        this.target[2] + this.distance * Math.sin(this.polar) * Math.cos(this.azimuth),
      ];
    }

    _bindInteraction() {
      const c = this.canvas;
      c.addEventListener("contextmenu", e => e.preventDefault());

      c.addEventListener("wheel", e => {
        e.preventDefault();
        this.zoomBy(e.deltaY > 0 ? 1.08 : 0.92);
      }, { passive: false });

      c.addEventListener("pointerdown", e => {
        c.setPointerCapture(e.pointerId);
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        this.autoOrbit = false;
        this.flight = null;
        if (this.pointers.size === 1) {
          this.lastDrag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
        } else if (this.pointers.size === 2) {
          const pts = [...this.pointers.values()];
          this.lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        }
      });

      c.addEventListener("pointermove", e => {
        if (!this.pointers.has(e.pointerId)) return;
        this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (this.pointers.size === 2) {
          const pts = [...this.pointers.values()];
          const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
          if (this.lastPinchDist > 10) {
            this.zoomBy(this.lastPinchDist / Math.max(dist, 10));
          }
          this.lastPinchDist = dist;
          return;
        }

        if (this.pointers.size === 1 && this.lastDrag) {
          const dx = e.clientX - this.lastDrag.x;
          const dy = e.clientY - this.lastDrag.y;
          this.lastDrag.x = e.clientX;
          this.lastDrag.y = e.clientY;

          if (this.mode === "interior") {
            // Look around inside room
            const vx = this.target[0] - this.eye[0];
            const vy = this.target[1] - this.eye[1];
            const vz = this.target[2] - this.eye[2];
            const dist = Math.hypot(vx, vy, vz) || 240;
            let yaw = Math.atan2(vx, vz) - dx * 0.0045;
            let pitch = Math.asin(Math.max(-0.85, Math.min(0.85, vy / dist))) - dy * 0.0035;
            pitch = Math.max(-0.65, Math.min(0.55, pitch));
            this.target = [
              this.eye[0] + dist * Math.cos(pitch) * Math.sin(yaw),
              this.eye[1] + dist * Math.sin(pitch),
              this.eye[2] + dist * Math.cos(pitch) * Math.cos(yaw),
            ];
          } else if (this.lastDrag.pan) {
            const panScale = this.distance * 0.0012;
            const cosA = Math.cos(this.azimuth), sinA = Math.sin(this.azimuth);
            this.target[0] -= (dx * cosA + dy * sinA) * panScale;
            this.target[2] -= (-dx * sinA + dy * cosA) * panScale;
            this._syncOrbitEye();
          } else {
            this.azimuth -= dx * 0.0055;
            this.polar = Math.max(0.12, Math.min(1.42, this.polar - dy * 0.0048));
            this._syncOrbitEye();
          }
        }
      });

      const endPointer = e => {
        this.pointers.delete(e.pointerId);
        if (this.pointers.size < 2) this.lastPinchDist = 0;
        if (this.pointers.size === 0) this.lastDrag = null;
      };
      c.addEventListener("pointerup", endPointer);
      c.addEventListener("pointercancel", endPointer);
    }

    project(worldX, worldY, worldZ) {
      const m = this.vpMatrix;
      const cx = m[0] * worldX + m[4] * worldY + m[8] * worldZ + m[12];
      const cy = m[1] * worldX + m[5] * worldY + m[9] * worldZ + m[13];
      const cz = m[2] * worldX + m[6] * worldY + m[10] * worldZ + m[14];
      const cw = m[3] * worldX + m[7] * worldY + m[11] * worldZ + m[15];
      if (cw <= 0.1) return null;
      const ndcX = cx / cw;
      const ndcY = cy / cw;
      const ndcZ = cz / cw;
      if (ndcX < -1.15 || ndcX > 1.15 || ndcY < -1.15 || ndcY > 1.15 || ndcZ > 1.0) return null;
      return {
        x: (ndcX * 0.5 + 0.5) * this.canvas.clientWidth,
        y: (1.0 - (ndcY * 0.5 + 0.5)) * this.canvas.clientHeight,
        depth: cw,
      };
    }

    render(now) {
      const gl = this.gl;
      if (!gl || !this.scene) return;

      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
      const ch = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
      if (this.canvas.width !== cw || this.canvas.height !== ch) {
        this.canvas.width = cw;
        this.canvas.height = ch;
      }
      gl.viewport(0, 0, cw, ch);

      // Update camera flight or auto-orbit
      if (this.flight) {
        const t = Math.min(1, (now - this.flight.startTime) / this.flight.duration);
        const ease = 1 - Math.pow(1 - t, 3);
        for (let i = 0; i < 3; i++) {
          this.eye[i] = this.flight.fromEye[i] + (this.flight.toEye[i] - this.flight.fromEye[i]) * ease;
          this.target[i] = this.flight.fromTarget[i] + (this.flight.toTarget[i] - this.flight.fromTarget[i]) * ease;
        }
        this.fov = this.flight.fromFov + (this.flight.toFov - this.flight.fromFov) * ease;
        if (t >= 1) {
          this.mode = this.flight.toMode;
          this.flight = null;
        }
      } else if (this.autoOrbit && this.mode === "orbit") {
        this.azimuth += 0.0022;
        this._syncOrbitEye();
      }

      // Smoothly interpolate the 6 point light intensities
      const lPos = new Float32Array(18);
      const lCol = new Float32Array(18);
      const lInt = new Float32Array(6);
      for (let i = 0; i < 6; i++) {
        const anchor = this.lightAnchors[i];
        if (!anchor) continue;
        const targetVal = this.lightTargets[anchor.key] || 0.0;
        const curVal = this.lightCurrent[anchor.key] || 0.0;
        const nextVal = curVal + (targetVal - curVal) * 0.14;
        this.lightCurrent[anchor.key] = nextVal;
        lPos[i * 3] = anchor.x;
        lPos[i * 3 + 1] = anchor.lightZ || 160.0;
        lPos[i * 3 + 2] = anchor.y;
        const [lr, lg, lb] = hexToRgb(anchor.lightColor || "#ffc86b");
        lCol[i * 3] = lr;
        lCol[i * 3 + 1] = lg;
        lCol[i * 3 + 2] = lb;
        lInt[i] = nextVal;
      }

      const proj = mat4Perspective((this.fov * Math.PI) / 180, cw / ch, 12.0, 4800.0);
      const view = mat4LookAt(this.eye, this.target, [0, 1, 0]);
      this.vpMatrix = mat4Multiply(proj, view);

      gl.clearColor(0.045, 0.052, 0.070, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);

      gl.useProgram(this.prog);
      gl.uniformMatrix4fv(this.locs.uVP, false, this.vpMatrix);
      gl.uniform3f(this.locs.uEye, this.eye[0], this.eye[1], this.eye[2]);
      gl.uniform3fv(this.locs.uLightPos, lPos);
      gl.uniform3fv(this.locs.uLightCol, lCol);
      gl.uniform1fv(this.locs.uLightInt, lInt);

      const drawBatch = (b, clampY) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, b.vbo);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.ibo);
        const stride = 12 * 4;
        gl.enableVertexAttribArray(this.locs.aPos);
        gl.vertexAttribPointer(this.locs.aPos, 3, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(this.locs.aNormal);
        gl.vertexAttribPointer(this.locs.aNormal, 3, gl.FLOAT, false, stride, 12);
        gl.enableVertexAttribArray(this.locs.aColor);
        gl.vertexAttribPointer(this.locs.aColor, 3, gl.FLOAT, false, stride, 24);
        gl.enableVertexAttribArray(this.locs.aMaterial);
        gl.vertexAttribPointer(this.locs.aMaterial, 3, gl.FLOAT, false, stride, 36);
        gl.uniform1f(this.locs.uOpacity, b.opacity);
        if (clampY !== null) {
          gl.uniform3f(this.locs.uScaleY, clampY, 1.0, 0.0);
        } else {
          gl.uniform3f(this.locs.uScaleY, 999.0, 0.0, 0.0);
        }
        gl.drawElements(gl.TRIANGLES, b.count, gl.UNSIGNED_SHORT, 0);
      };

      // Draw static opaque batches (floors + furniture)
      for (const b of this.opaqueBatches) {
        drawBatch(b, null);
      }

      // Draw walls with smooth smart directional cutaway
      const aptCx = this.scene.bounds.centerX;
      const aptCz = this.scene.bounds.centerY;
      const camVecX = this.eye[0] - aptCx;
      const camVecZ = this.eye[2] - aptCz;
      for (const w of this.wallDefs) {
        let targetH = w.height;
        if (this.cutawayEnabled && this.mode === "orbit") {
          const wVecX = w.midX - aptCx;
          const wVecZ = w.midZ - aptCz;
          const facingDot = wVecX * camVecX + wVecZ * camVecZ;
          if (w.exterior && facingDot > -12000) {
            targetH = 38.0;
          } else if (!w.exterior) {
            targetH = 92.0;
          }
        }
        const curH = this.wallCuts[w.id] ?? w.height;
        const nextH = curH + (targetH - curH) * 0.16;
        this.wallCuts[w.id] = nextH;
        drawBatch(w.batch, nextH);
      }

      // Draw transparent glass panes & balcony railings
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      for (const b of this.glassBatches) {
        drawBatch(b, null);
      }
      gl.depthMask(true);
      gl.disable(gl.BLEND);

      if (this.onPinProjected) this.onPinProjected();
      if (this.onRoomProjected) this.onRoomProjected();
    }
  }

  // =========================================================================
  //  Main Application Controller
  // =========================================================================
  const API_BASE = window.location.hostname === "altayatik.com"
    ? "https://houseai-relay.altayatik.workers.dev"
    : "";

  const App = {
    csrfToken: "",
    sessionToken: "",
    authMode: "otp",
    codeDigits: "",
    scene: null,
    state: null,
    weather: null,
    energy: null,
    chatMessages: [],
    pinFilter: "all",
    selectedPinKey: null,
    engine: null,
    pinElements: new Map(),
    roomElements: new Map(),
    pollTimer: null,

    async init() {
      try {
        this.sessionToken = sessionStorage.getItem("hv_sess_tok") || "";
      } catch (_e) {}
      this._bindAuthUI();
      this._bindDashboardUI();
      this._updateClock();
      setInterval(() => this._updateClock(), 1000);
      await this.checkAuthStatus();
    },

    _updateClock() {
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
      const dateStr = now.toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
      const hc = document.getElementById("hud-clock");
      const hd = document.getElementById("hud-date");
      const dt = document.getElementById("dimmer-time");
      const dd = document.getElementById("dimmer-date");
      if (hc) hc.textContent = timeStr;
      if (hd) hd.textContent = dateStr;
      if (dt) dt.textContent = timeStr.replace(/\s?[AP]M/i, "");
      if (dd) dd.textContent = dateStr;
    },

    showToast(msg) {
      const toast = document.getElementById("toast-banner");
      if (!toast) return;
      toast.textContent = msg;
      toast.classList.remove("hidden");
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => toast.classList.add("hidden"), 3800);
    },

    async apiFetch(path, options = {}) {
      const headers = Object.assign({ "Accept": "application/json" }, options.headers || {});
      if (this.sessionToken) {
        headers["X-HV-Session"] = this.sessionToken;
      }
      if (options.body) {
        headers["Content-Type"] = "application/json";
        if (this.csrfToken) headers["X-CSRF-Token"] = this.csrfToken;
      }
      const resp = await fetch(API_BASE + path, {
        method: options.method || "GET",
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined,
        credentials: API_BASE ? "omit" : "same-origin",
      });
      const data = await resp.json().catch(() => ({}));
      if (resp.status === 401 && !path.includes("/auth/")) {
        this.sessionToken = "";
        try { sessionStorage.removeItem("hv_sess_tok"); } catch (_e) {}
        this._showLoginScreen();
        throw new Error(data.message || "Session expired.");
      }
      if (!resp.ok) {
        throw new Error(data.message || data.error || `Request failed (${resp.status})`);
      }
      return data;
    },

    async checkAuthStatus() {
      try {
        const st = await this.apiFetch("/homeview/api/auth/status");
        this._configureAuthScreen(st);
        if (st.authenticated) {
          this.csrfToken = st.csrf_token || "";
          await this._enterDashboard();
        } else {
          this._showLoginScreen();
        }
      } catch (err) {
        this._showLoginScreen(err.message);
      }
    },

    _configureAuthScreen(st) {
      const badge = document.getElementById("auth-network-badge");
      const badgeText = document.getElementById("auth-network-text");
      const tabs = document.getElementById("auth-mode-tabs");
      const passGroup = document.getElementById("passphrase-group");

      const isLocal = st.network === "local_wifi";
      badge.className = "hv-net-badge " + (isLocal ? "local" : "remote");
      badgeText.textContent = isLocal
        ? "Local Wi-Fi Detected · Single-Use One-Time Passcode Allowed"
        : "Remote Internet Access · 2FA Authenticator Code Required";

      if (isLocal) {
        tabs.classList.remove("hidden");
        this._setAuthMode("otp", st.passphrase_required);
      } else {
        tabs.classList.add("hidden");
        this._setAuthMode("2fa", st.passphrase_required);
      }
      if (passGroup) {
        passGroup.classList.toggle("hidden", !(this.authMode === "2fa" && st.passphrase_required));
      }
    },

    _setAuthMode(mode, passRequired) {
      this.authMode = mode;
      this.codeDigits = "";
      this._renderCodeSlots();
      const tabOtp = document.getElementById("tab-auth-otp");
      const tab2fa = document.getElementById("tab-auth-2fa");
      const desc = document.getElementById("auth-instructions");
      const passGroup = document.getElementById("passphrase-group");

      if (tabOtp && tab2fa) {
        tabOtp.classList.toggle("active", mode === "otp");
        tab2fa.classList.toggle("active", mode === "2fa");
      }
      if (desc) {
        desc.textContent = mode === "otp"
          ? "Enter the single-use 6-digit One-Time Passcode generated on your Mac mini (run `houseai-otp`)."
          : "Enter the 6-digit 2FA code from your Authenticator app to unlock remote access.";
      }
      if (passGroup && passRequired !== undefined) {
        passGroup.classList.toggle("hidden", !(mode === "2fa" && passRequired));
      }
    },

    _renderCodeSlots() {
      const slots = document.querySelectorAll("#code-boxes .hv-code-slot");
      slots.forEach((slot, idx) => {
        const ch = this.codeDigits[idx] || "";
        slot.textContent = ch;
        slot.classList.toggle("filled", Boolean(ch));
      });
    },

    _bindAuthUI() {
      const form = document.getElementById("auth-form");
      const errBox = document.getElementById("auth-error");
      const tabOtp = document.getElementById("tab-auth-otp");
      const tab2fa = document.getElementById("tab-auth-2fa");

      if (tabOtp) tabOtp.addEventListener("click", () => this._setAuthMode("otp"));
      if (tab2fa) tab2fa.addEventListener("click", () => this._setAuthMode("2fa"));

      document.querySelectorAll(".hv-key").forEach(btn => {
        btn.addEventListener("click", () => {
          errBox.classList.add("hidden");
          const k = btn.dataset.key;
          if (k === "clear") {
            this.codeDigits = "";
          } else if (k === "back") {
            this.codeDigits = this.codeDigits.slice(0, -1);
          } else if (this.codeDigits.length < 6 && /^\d$/.test(k)) {
            this.codeDigits += k;
          }
          this._renderCodeSlots();
          if (this.codeDigits.length === 6) {
            form.requestSubmit();
          }
        });
      });

      window.addEventListener("keydown", e => {
        const authScreen = document.getElementById("auth-screen");
        if (!authScreen || authScreen.classList.contains("hidden")) return;
        if (document.activeElement && document.activeElement.id === "auth-passphrase") return;
        if (/^\d$/.test(e.key) && this.codeDigits.length < 6) {
          errBox.classList.add("hidden");
          this.codeDigits += e.key;
          this._renderCodeSlots();
          if (this.codeDigits.length === 6) form.requestSubmit();
        } else if (e.key === "Backspace") {
          this.codeDigits = this.codeDigits.slice(0, -1);
          this._renderCodeSlots();
        }
      });

      form.addEventListener("submit", async e => {
        e.preventDefault();
        errBox.classList.add("hidden");
        const submitBtn = document.getElementById("auth-submit-btn");
        const passphrase = (document.getElementById("auth-passphrase")?.value || "").trim();
        if (this.codeDigits.length !== 6) {
          errBox.textContent = "Please enter all 6 digits.";
          errBox.classList.remove("hidden");
          return;
        }
        submitBtn.disabled = true;
        submitBtn.textContent = "Verifying…";
        try {
          const res = await this.apiFetch("/homeview/api/auth/login", {
            method: "POST",
            body: { mode: this.authMode, code: this.codeDigits, passphrase },
          });
          this.csrfToken = res.csrf_token || "";
          if (res.session_token) {
            this.sessionToken = res.session_token;
            try { sessionStorage.setItem("hv_sess_tok", res.session_token); } catch (_e) {}
          }
          this.codeDigits = "";
          this._renderCodeSlots();
          await this._enterDashboard();
        } catch (err) {
          this.codeDigits = "";
          this._renderCodeSlots();
          errBox.textContent = err.message;
          errBox.classList.remove("hidden");
        } finally {
          submitBtn.disabled = false;
          submitBtn.textContent = "Unlock HomeView 3D";
        }
      });
    },

    _showLoginScreen(msg) {
      clearInterval(this.pollTimer);
      document.getElementById("dashboard-screen").classList.add("hidden");
      document.getElementById("auth-screen").classList.remove("hidden");
      if (msg) {
        const errBox = document.getElementById("auth-error");
        errBox.textContent = msg;
        errBox.classList.remove("hidden");
      }
    },

    async _enterDashboard() {
      document.getElementById("auth-screen").classList.add("hidden");
      document.getElementById("dashboard-screen").classList.remove("hidden");

      if (!this.engine) {
        const canvas = document.getElementById("sh3d-canvas");
        this.engine = new Apartment3DEngine(
          canvas,
          () => this._updateProjectedPins(),
          () => this._updateProjectedRooms()
        );
        const loop = now => {
          this.engine.render(now);
          requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
      }

      if (!this.scene) {
        const sceneData = await this.apiFetch("/homeview/api/scene");
        this.scene = sceneData;
        this.engine.loadScene(sceneData);
        this._buildPinDOM();
        this._buildRoomLabelsDOM();
      }

      await Promise.all([
        this.refreshState(),
        this.refreshWeather(),
        this.refreshEnergy(),
        this.refreshChat(),
      ]);

      clearInterval(this.pollTimer);
      this.pollTimer = setInterval(() => this.refreshState(), 4000);
    },

    _buildRoomLabelsDOM() {
      const container = document.getElementById("rooms-layer");
      container.replaceChildren();
      this.roomElements.clear();
      for (const rm of this.scene.rooms || []) {
        const badge = el("div", { className: "hv-room-tag", text: rm.name });
        container.appendChild(badge);
        this.roomElements.set(rm.id, { rm, el: badge });
      }
    },

    _buildPinDOM() {
      const container = document.getElementById("pins-layer");
      container.replaceChildren();
      this.pinElements.clear();
      const iconMap = {
        lightbulb: "💡", lamp: "🛋", thermostat: "🌡", wind: "🌀",
        droplet: "💧", sparkles: "✨", washer: "🫧", dryer: "♨",
        tv: "📺", speaker: "🔊", coffee: "☕", battery: "🔋",
        disc: "🤖", door: "🚪", window: "🪟", lock: "🔐",
        motion: "📡", button: "🔘", camera: "📷", hub: "📶",
      };
      for (const dev of this.scene.smartDevices || []) {
        const stateSpan = el("span", { className: "hv-pin-state", text: "" });
        const pin = el("button", {
          type: "button",
          className: "hv-pin",
          onClick: e => {
            e.stopPropagation();
            this._handlePinClick(dev);
          },
        }, [
          el("span", { text: iconMap[dev.icon] || "•" }),
          el("span", { text: dev.shortLabel }),
          stateSpan,
        ]);
        container.appendChild(pin);
        this.pinElements.set(dev.id, { dev, el: pin, stateSpan });
      }
    },

    _matchesPinFilter(dev) {
      if (this.pinFilter === "none") return false;
      if (this.pinFilter === "all") return true;
      if (this.pinFilter === "light") return dev.category === "light";
      if (this.pinFilter === "appliance") {
        return ["climate", "air_purifier", "aquatru", "dishwasher", "laundry", "tv", "media", "speaker", "appliance"].includes(dev.category);
      }
      if (this.pinFilter === "sensor") {
        return ["contact", "lock", "motion", "button", "camera", "hub", "power", "sensor_device"].includes(dev.category);
      }
      return true;
    },

    _updateProjectedRooms() {
      if (!this.engine || !this.scene) return;
      const showLabels = this.engine.mode === "orbit" && this.pinFilter !== "none";
      for (const { rm, el: node } of this.roomElements.values()) {
        if (!showLabels) {
          node.style.display = "none";
          continue;
        }
        const pt = this.engine.project(rm.center[0], 8.0, rm.center[1]);
        if (!pt) {
          node.style.display = "none";
          continue;
        }
        node.style.display = "block";
        node.style.left = `${pt.x.toFixed(1)}px`;
        node.style.top = `${pt.y.toFixed(1)}px`;
      }
    },

    _updateProjectedPins() {
      if (!this.engine || !this.scene) return;
      for (const { dev, el: node } of this.pinElements.values()) {
        if (!this._matchesPinFilter(dev)) {
          node.style.display = "none";
          continue;
        }
        const pt = this.engine.project(dev.x, dev.pinZ || (dev.z + dev.h + 16), dev.y);
        if (!pt) {
          node.style.display = "none";
          continue;
        }
        node.style.display = "inline-flex";
        node.style.left = `${pt.x.toFixed(1)}px`;
        node.style.top = `${pt.y.toFixed(1)}px`;
        node.style.zIndex = String(Math.max(1, Math.min(25, Math.round(3000 - pt.depth))));
      }
    },

    async _handlePinClick(dev) {
      if (dev.category === "light") {
        // Immediate light toggle on 3D lamp tap!
        await this.sendAction({ domain: "lights", action: "toggle", target: dev.target });
        return;
      }
      this.engine.focusPoint(dev.x, dev.y, dev.z);
      this.selectedPinKey = dev.key;
      this._renderDevicePopover(dev);
    },

    _renderDevicePopover(dev) {
      const pop = document.getElementById("device-popover");
      const title = document.getElementById("popover-title");
      const room = document.getElementById("popover-room");
      const body = document.getElementById("popover-body");
      if (!pop || !this.state) return;

      room.textContent = (dev.roomArea || dev.room || "APARTMENT").toUpperCase();
      title.textContent = dev.label;
      body.replaceChildren();

      if (dev.category === "climate") {
        const c = this.state.climate;
        body.appendChild(el("div", { className: "hv-subtext", text: `Current ${c.current_temperature}°F · Target ${c.target_temperature}°F (${c.hvac_action})` }));
        body.appendChild(el("div", { className: "hv-btn-row-3" }, [
          el("button", { className: "hv-chip-btn", text: "−1°F", onClick: () => this.adjustTemp(-1) }),
          el("button", { className: "hv-chip-btn", text: "+1°F", onClick: () => this.adjustTemp(1) }),
          el("button", { className: "hv-chip-btn", text: "71°F", onClick: () => this.sendAction({ domain: "climate", action: "set", temperature_f: 71 }) }),
        ]));
      } else if (dev.category === "air_purifier") {
        const ap = this.state.air_purifier || {};
        body.appendChild(el("div", { className: "hv-subtext", text: `State: ${(ap.state || "off").toUpperCase()} · Speed: ${ap.percentage ?? 0}%` }));
        body.appendChild(el("div", { className: "hv-btn-row-3" }, [
          el("button", { className: "hv-chip-btn", text: "Turn On", onClick: () => this.sendAction({ domain: "air_purifier", action: "on" }) }),
          el("button", { className: "hv-chip-btn", text: "50% Speed", onClick: () => this.sendAction({ domain: "air_purifier", action: "speed", percentage: 50 }) }),
          el("button", { className: "hv-chip-btn", text: "Turn Off", onClick: () => this.sendAction({ domain: "air_purifier", action: "off" }) }),
        ]));
      } else if (dev.category === "dishwasher") {
        const dw = this.state.dishwasher_status || {};
        body.appendChild(el("div", { className: "hv-subtext", text: `Status: ${dw.status || "Idle"}` }));
        body.appendChild(el("div", { className: "hv-btn-row-3" }, [
          el("button", { className: "hv-chip-btn", text: "✨ Clean", onClick: () => this.sendAction({ domain: "dishwasher", action: "clean" }) }),
          el("button", { className: "hv-chip-btn", text: "🍽 Dirty", onClick: () => this.sendAction({ domain: "dishwasher", action: "dirty" }) }),
          el("button", { className: "hv-chip-btn", text: "✓ Emptied", onClick: () => this.sendAction({ domain: "dishwasher", action: "idle" }) }),
        ]));
      } else if (dev.category === "aquatru") {
        const aq = this.state.aquatru || {};
        body.appendChild(el("div", { className: "hv-subtext", text: `Clean TDS: ${aq.clean_tds_ppm} ppm · Tap TDS: ${aq.tap_tds_ppm} ppm` }));
        body.appendChild(el("div", { className: "hv-subtext", text: `Filters: Pre ${aq.pre_filter_pct}% · VOC ${aq.voc_filter_pct}% · RO ${aq.ro_filter_pct}%` }));
      } else if (dev.category === "laundry") {
        const items = this.state.laundry || [];
        const match = items.find(i => i.kind === dev.target) || items[0];
        body.appendChild(el("div", { className: "hv-subtext", text: match ? `${match.name}: ${match.status}${match.remaining_minutes ? ` · ${match.remaining_minutes}m left` : ""}` : "Idle" }));
      } else if (dev.category === "tv") {
        body.appendChild(el("div", { className: "hv-btn-row-3" }, [
          el("button", { className: "hv-chip-btn", text: "Power On", onClick: () => this.sendAction({ domain: "tv", action: "on", target: dev.target }) }),
          el("button", { className: "hv-chip-btn", text: "⏯ Play", onClick: () => this.sendAction({ domain: "tv", action: "play_pause", target: dev.target }) }),
          el("button", { className: "hv-chip-btn", text: "Power Off", onClick: () => this.sendAction({ domain: "tv", action: "off", target: dev.target }) }),
        ]));
      } else if (dev.category === "speaker") {
        const sp = this.state.speaker || {};
        body.appendChild(el("div", { className: "hv-subtext", text: `${sp.device_name || "Speaker"} · ${sp.connected ? "Connected" : "Standby"}` }));
        body.appendChild(el("div", { className: "hv-btn-row-wrap" }, [
          el("button", { className: "hv-chip-btn", text: "🔊 Dinner is ready", onClick: () => this.sendAction({ domain: "speaker", action: "announce", message: "Dinner is ready." }) }),
          el("button", { className: "hv-chip-btn", text: "🔊 Coffee is ready", onClick: () => this.sendAction({ domain: "speaker", action: "announce", message: "Fresh coffee is ready in the kitchen." }) }),
        ]));
      } else {
        const matchingSensors = (this.state.sensors || []).filter(s => {
          const hay = (s.name + " " + s.id).toLowerCase();
          return hay.includes(dev.target.toLowerCase()) || hay.includes(dev.shortLabel.toLowerCase());
        });
        if (matchingSensors.length) {
          for (const s of matchingSensors.slice(0, 4)) {
            body.appendChild(el("div", { className: "hv-row-item" }, [
              el("span", { text: s.name }),
              el("strong", { text: s.display }),
            ]));
          }
        } else {
          body.appendChild(el("div", { className: "hv-subtext", text: `Positioned in ${dev.roomArea} · Connected to Home Assistant` }));
        }
      }

      pop.classList.remove("hidden");
    },

    async refreshState() {
      try {
        const st = await this.apiFetch("/homeview/api/state");
        this.state = st;
        if (this.engine) {
          this.engine.updateLightStates(st.lights || []);
        }
        this._renderAllState(st);
      } catch (_err) {
        // Handled by apiFetch
      }
    },

    async refreshWeather() {
      try {
        const w = await this.apiFetch("/homeview/api/weather");
        this.weather = w;
        this._renderWeather(w);
      } catch (_err) {}
    },

    async refreshEnergy() {
      try {
        const en = await this.apiFetch("/homeview/api/energy");
        this.energy = en;
        this._renderEnergy(en);
      } catch (_err) {}
    },

    async refreshChat() {
      try {
        const ch = await this.apiFetch("/homeview/api/chat");
        this.chatMessages = ch.messages || [];
        this._renderChat();
      } catch (_err) {}
    },

    async sendAction(payload) {
      // Optimistic 3D light feedback
      if (payload.domain === "lights" && this.state && this.engine) {
        for (const l of this.state.lights || []) {
          if (payload.target === "all" || l.id === payload.target || l.room === payload.target) {
            l.is_on = payload.action === "toggle" ? !l.is_on : payload.action === "on";
          }
        }
        this.engine.updateLightStates(this.state.lights);
        this._renderAllState(this.state);
      }
      try {
        const res = await this.apiFetch("/homeview/api/action", {
          method: "POST",
          body: payload,
        });
        if (res.message) this.showToast(res.message);
        if (res.state) {
          this.state = res.state;
          if (this.engine) this.engine.updateLightStates(res.state.lights || []);
          this._renderAllState(res.state);
        }
      } catch (err) {
        this.showToast(err.message);
        await this.refreshState();
      }
    },

    adjustTemp(delta) {
      if (!this.state) return;
      const cur = this.state.climate?.target_temperature || 71;
      const next = Math.max(65, Math.min(80, cur + delta));
      this.sendAction({ domain: "climate", action: "set", temperature_f: next });
    },

    _renderWeather(w) {
      if (!w) return;
      const icons = {
        "sun": "☀", "moon": "🌙", "cloud-sun": "⛅", "cloud-moon": "☁",
        "cloud": "☁", "fog": "🌫", "rain": "🌧", "snow": "❄", "storm": "⛈",
      };
      document.getElementById("weather-icon").textContent = icons[w.icon] || "⛅";
      document.getElementById("weather-temp").textContent = `${w.temp_f}°F`;
      document.getElementById("weather-cond").textContent = w.condition;
      document.getElementById("weather-sub").textContent =
        `${w.location} · H:${w.high_f}° L:${w.low_f}° · Wind ${w.wind_mph} mph`;

      const modalBody = document.getElementById("weather-modal-body");
      const modalTitle = document.getElementById("weather-modal-title");
      if (modalBody && modalTitle) {
        modalTitle.textContent = `${w.temp_f}°F · ${w.condition}`;
        modalBody.replaceChildren(
          el("div", { className: "hv-subtext", text: `Feels like ${w.feels_like_f}°F · Humidity ${w.humidity_pct}% · Wind ${w.wind_mph} mph · High ${w.high_f}°F / Low ${w.low_f}°F` }),
          el("div", { className: "hv-btn-row-wrap" }, (w.hourly || []).map(h =>
            el("div", { className: "hv-row-item", style: "flex:1;min-width:78px;flex-direction:column;gap:2px;" }, [
              el("span", { className: "hv-card-eyebrow", text: h.time }),
              el("span", { text: icons[h.icon] || "⛅" }),
              el("strong", { text: `${h.temp_f}°` }),
            ])
          ))
        );
      }
    },

    _renderAllState(st) {
      if (!st) return;

      // 1. Mode badge
      const badge = document.getElementById("hud-mode-badge");
      if (st.kill_switch) {
        badge.className = "hv-badge hv-badge-killed";
        badge.textContent = "KILLED";
      } else if (st.practice_mode) {
        badge.className = "hv-badge hv-badge-practice";
        badge.textContent = "PRACTICE";
      } else {
        badge.className = "hv-badge hv-badge-live";
        badge.textContent = "LIVE";
      }

      // 2. Update 3D Pin Badges & Glows
      const lightsById = Object.fromEntries((st.lights || []).map(l => [l.id, l]));
      for (const { dev, el: pinNode, stateSpan } of this.pinElements.values()) {
        pinNode.classList.remove("is-on", "is-active", "is-alert");
        stateSpan.textContent = "";
        if (dev.category === "light") {
          const l = lightsById[dev.key];
          const on = Boolean(l && l.is_on);
          pinNode.classList.toggle("is-on", on);
          stateSpan.textContent = on ? "ON" : "OFF";
        } else if (dev.category === "climate" && st.climate) {
          pinNode.classList.add("is-active");
          stateSpan.textContent = `${st.climate.current_temperature}°→${st.climate.target_temperature}°`;
        } else if (dev.category === "air_purifier" && st.air_purifier) {
          const on = Boolean(st.air_purifier.is_on);
          pinNode.classList.toggle("is-active", on);
          stateSpan.textContent = on ? `${st.air_purifier.percentage ?? 100}%` : "OFF";
        } else if (dev.category === "dishwasher" && st.dishwasher_status) {
          const status = st.dishwasher_status.status || "Idle";
          pinNode.classList.toggle("is-active", status === "Running" || status === "Clean");
          stateSpan.textContent = status;
        } else if (dev.category === "aquatru" && st.aquatru) {
          const alert = Boolean(st.aquatru.needs_refill_nag || st.aquatru.refill_needed);
          pinNode.classList.toggle("is-alert", alert);
          pinNode.classList.toggle("is-active", !alert);
          stateSpan.textContent = alert ? "REFILL" : `${st.aquatru.clean_tds_ppm} ppm`;
        } else if (dev.category === "laundry") {
          const m = (st.laundry || []).find(x => x.kind === dev.target);
          if (m) {
            pinNode.classList.toggle("is-active", Boolean(m.running || m.complete));
            stateSpan.textContent = m.status;
          }
        } else {
          const sensor = (st.sensors || []).find(s => {
            const hay = (s.name + " " + s.id).toLowerCase();
            return hay.includes(dev.target.toLowerCase());
          });
          if (sensor) {
            pinNode.classList.toggle("is-alert", Boolean(sensor.is_alert));
            stateSpan.textContent = sensor.display;
          }
        }
      }

      // 3. Quick Room Dock & Lights Grid
      const lights = st.lights || [];
      const onCount = lights.filter(l => l.is_on).length;
      document.getElementById("lights-summary-title").textContent = `${onCount} of ${lights.length} On`;

      const roomCounts = {
        "living room": [0, 0, "dock-room-living", "dock-cnt-living"],
        "bedroom": [0, 0, "dock-room-bedroom", "dock-cnt-bedroom"],
        "den": [0, 0, "dock-room-den", "dock-cnt-den"],
        "hallway": [0, 0, "dock-room-hallway", "dock-cnt-hallway"],
      };
      for (const l of lights) {
        const rc = roomCounts[l.room];
        if (rc) {
          rc[1] += 1;
          if (l.is_on) rc[0] += 1;
        }
      }
      for (const [, [lit, tot, btnId, cntId]] of Object.entries(roomCounts)) {
        const b = document.getElementById(btnId);
        const c = document.getElementById(cntId);
        if (b) b.classList.toggle("lit", lit > 0);
        if (c) c.textContent = `${lit}/${tot || 1}`;
      }

      const lGrid = document.getElementById("lights-grid");
      lGrid.replaceChildren(...lights.map(l =>
        el("button", {
          type: "button",
          className: "hv-light-tile" + (l.is_on ? " on" : ""),
          onClick: () => this.sendAction({ domain: "lights", action: "toggle", target: l.id }),
        }, [
          el("span", { className: "hv-light-name", text: l.name }),
          el("span", { className: "hv-light-meta", text: `${l.room.toUpperCase()} · ${l.is_on ? "ON" : "OFF"}` }),
        ])
      ));

      // 4. Climate
      if (st.climate) {
        const c = st.climate;
        document.getElementById("hud-indoor-temp").textContent = String(c.current_temperature);
        document.getElementById("hud-indoor-sub").textContent = `Target ${c.target_temperature}°F · ${c.hvac_action}`;
        document.getElementById("climate-card-title").textContent = `${c.current_temperature}°F Indoor`;
        document.getElementById("climate-target-val").textContent = String(c.target_temperature);
        document.getElementById("climate-hvac-badge").textContent = c.hvac_action.toUpperCase();
        document.querySelectorAll("[data-climate-mode]").forEach(btn => {
          btn.classList.toggle("active", btn.dataset.climateMode === c.state);
        });
      }

      // 5. Air Purifier
      if (st.air_purifier) {
        const ap = st.air_purifier;
        document.getElementById("purifier-title").textContent = ap.is_on ? `On (${ap.percentage ?? 100}%)` : "Off";
        document.getElementById("purifier-speed-val").textContent = `${ap.percentage ?? 0}%`;
        const slider = document.getElementById("purifier-slider");
        if (slider && document.activeElement !== slider) {
          slider.value = String(ap.percentage ?? 0);
        }
      }

      // 6. Alerts Section
      const alertsBox = document.getElementById("alerts-section");
      const sensorAlerts = (st.sensors || []).filter(s => s.is_alert);
      const activeRems = st.reminders?.active || [];
      const waterAlert = st.aquatru?.needs_refill_nag;
      if (sensorAlerts.length || activeRems.length || waterAlert) {
        alertsBox.classList.remove("hidden");
        const items = [];
        for (const sa of sensorAlerts) {
          items.push(el("div", { className: "hv-row-item alert" }, [
            el("span", { text: `⚠ ${sa.name}` }),
            el("strong", { text: sa.display }),
          ]));
        }
        if (waterAlert) {
          items.push(el("div", { className: "hv-row-item alert" }, [
            el("span", { text: "💧 AquaTru Water Needs Attention" }),
            el("strong", { text: (st.aquatru.reasons || ["Refill"]).join(", ") }),
          ]));
        }
        for (const r of activeRems) {
          items.push(el("div", { className: "hv-row-item alert" }, [
            el("span", { text: `🔔 ${r.name}` }),
            el("div", { className: "hv-btn-group" }, [
              el("button", { className: "hv-chip-btn", text: "Done", onClick: () => this.sendAction({ domain: "reminders", action: "done", name: r.id }) }),
              el("button", { className: "hv-chip-btn", text: "+30m", onClick: () => this.sendAction({ domain: "reminders", action: "push", name: r.id, minutes: 30 }) }),
            ]),
          ]));
        }
        alertsBox.replaceChildren(
          el("div", { className: "hv-card-eyebrow", text: "ACTIVE HOUSEHOLD ALERTS" }),
          el("div", { className: "hv-stack", style: "margin-top:8px;" }, items)
        );
      } else {
        alertsBox.classList.add("hidden");
      }

      // 7. Dishwasher, Laundry, AquaTru
      if (st.dishwasher_status) {
        document.getElementById("dishwasher-title").textContent = st.dishwasher_status.status || "Idle";
        document.getElementById("dishwasher-badge").textContent = st.dishwasher_status.running ? "Running" : st.dishwasher_status.clean_pending ? "Clean" : st.dishwasher_status.dirty_pending ? "Dirty" : "Ready";
      }
      const lList = document.getElementById("laundry-list");
      lList.replaceChildren(...((st.laundry || []).map(m =>
        el("div", { className: "hv-row-item" }, [
          el("span", { text: `${m.name}${m.cycle ? ` · ${m.cycle}` : ""}` }),
          el("strong", { text: `${m.status}${m.remaining_minutes ? ` (${m.remaining_minutes}m)` : ""}` }),
        ])
      )));

      if (st.aquatru) {
        const aq = st.aquatru;
        document.getElementById("aquatru-title").textContent = aq.needs_refill_nag ? "Refill Needed" : "Clean Water Ready";
        document.getElementById("aquatru-tds-badge").textContent = `${aq.clean_tds_ppm} ppm`;
        document.getElementById("aquatru-details").replaceChildren(
          el("div", { className: "hv-row-item" }, [el("span", { text: "Tap → Clean TDS" }), el("strong", { text: `${aq.tap_tds_ppm} ppm → ${aq.clean_tds_ppm} ppm` })]),
          el("div", { className: "hv-row-item" }, [el("span", { text: "Filter Life (Pre / VOC / RO)" }), el("strong", { text: `${aq.pre_filter_pct}% · ${aq.voc_filter_pct}% · ${aq.ro_filter_pct}%` })])
        );
      }

      // 8. Sensors list
      const sList = document.getElementById("sensors-list");
      sList.replaceChildren(...(st.sensors || []).map(s =>
        el("div", { className: "hv-row-item" + (s.is_alert ? " alert" : "") }, [
          el("span", { text: `${s.name} (${s.room})` }),
          el("strong", { text: s.display }),
        ])
      ));

      // 9. Groceries, Reminders/Chores, Packages, Safety
      const gList = document.getElementById("groceries-list");
      const groceries = st.groceries?.need || [];
      gList.replaceChildren(...(groceries.length ? groceries.map(item =>
        el("button", {
          type: "button",
          className: "hv-chip-btn",
          text: `✓ ${item}`,
          onClick: () => this.sendAction({ domain: "groceries", action: "bought", item }),
        })
      ) : [el("span", { className: "hv-subtext", text: "All stocked up!" })]));

      const rcList = document.getElementById("reminders-chores-list");
      const rcNodes = [];
      for (const c of st.chores || []) {
        rcNodes.push(el("div", { className: "hv-row-item" }, [
          el("span", { text: `${c.chore} (${c.who})` }),
          el("button", {
            type: "button",
            className: "hv-chip-btn",
            text: c.is_due ? "Mark Done" : `Due ${c.next}`,
            onClick: () => this.sendAction({ domain: "chores", action: "done", chore: c.chore }),
          }),
        ]));
      }
      for (const u of st.reminders?.upcoming_today || []) {
        rcNodes.push(el("div", { className: "hv-row-item" }, [
          el("span", { text: `⏰ ${u.name} at ${u.time}` }),
          el("button", {
            type: "button",
            className: "hv-chip-btn",
            text: "Done",
            onClick: () => this.sendAction({ domain: "reminders", action: "done", name: u.id }),
          }),
        ]));
      }
      rcList.replaceChildren(...(rcNodes.length ? rcNodes : [el("span", { className: "hv-subtext", text: "No pending chores or reminders." })]));

      const pList = document.getElementById("packages-list");
      const pkgs = st.packages || [];
      pList.replaceChildren(...(pkgs.length ? pkgs.map(p =>
        el("div", { className: "hv-row-item" }, [
          el("span", { text: `${p.label} (${p.carrier}): ${p.status}` }),
          el("button", {
            type: "button",
            className: "hv-chip-btn",
            text: "Picked Up",
            onClick: () => this.sendAction({ domain: "packages", action: "picked_up", tracking_number: p.number }),
          }),
        ])
      ) : [el("span", { className: "hv-subtext", text: "No active deliveries." })]));

      const killBtn = document.getElementById("btn-kill-switch");
      if (killBtn) {
        killBtn.textContent = st.kill_switch ? "▶ Resume Home Bot & Unlock" : "🛑 Stop Home Bot & Lock Devices";
      }
      const pracBtn = document.getElementById("btn-practice-mode");
      if (pracBtn) {
        pracBtn.textContent = `Practice Mode: ${st.practice_mode ? "ON" : "OFF"}`;
        pracBtn.classList.toggle("active", Boolean(st.practice_mode));
      }
    },

    _renderEnergy(en) {
      const container = document.getElementById("energy-list");
      if (!container || !en) return;
      const rows = en.rows || [];
      if (!rows.length) {
        container.replaceChildren(el("span", { className: "hv-subtext", text: en.message || "Energy data unavailable." }));
        return;
      }
      container.replaceChildren(...rows.map(r =>
        el("div", { className: "hv-row-item" }, [
          el("span", { text: r.name }),
          el("strong", { text: `${r.kwh !== null && r.kwh !== undefined ? r.kwh.toFixed(2) + " kWh" : "--"}${r.cost ? ` ($${r.cost.toFixed(2)})` : ""}` }),
        ])
      ));
    },

    _renderChat() {
      const log = document.getElementById("chat-log");
      if (!log) return;
      log.replaceChildren(...(this.chatMessages.slice(-8).map(m =>
        el("div", {
          className: "hv-chat-bubble " + (m.role === "user" ? "user" : "assistant"),
          text: m.text,
        })
      )));
      log.scrollTop = log.scrollHeight;
    },

    _bindDashboardUI() {
      // Camera viewpoint switcher
      document.querySelectorAll("#camera-bar .hv-cam-btn").forEach(btn => {
        btn.addEventListener("click", () => {
          document.querySelectorAll("#camera-bar .hv-cam-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          if (this.engine) {
            this.engine.setCamera(btn.dataset.cam, true);
            document.getElementById("btn-cutaway").classList.toggle("active", this.engine.cutawayEnabled);
          }
        });
      });

      // Cutaway, Auto-Orbit, Dimmer, Sidebar Toggle, Logout
      document.getElementById("btn-cutaway").addEventListener("click", e => {
        if (!this.engine) return;
        this.engine.cutawayEnabled = !this.engine.cutawayEnabled;
        e.currentTarget.classList.toggle("active", this.engine.cutawayEnabled);
      });

      document.getElementById("btn-autotour").addEventListener("click", e => {
        if (!this.engine) return;
        this.engine.autoOrbit = !this.engine.autoOrbit;
        e.currentTarget.classList.toggle("active", this.engine.autoOrbit);
      });

      const dimmer = document.getElementById("dimmer-overlay");
      document.getElementById("btn-dimmer").addEventListener("click", () => {
        dimmer.classList.remove("hidden");
      });
      dimmer.addEventListener("click", () => {
        dimmer.classList.add("hidden");
      });

      document.getElementById("btn-toggle-sidebar").addEventListener("click", e => {
        const sb = document.getElementById("control-sidebar");
        const hidden = sb.classList.toggle("hidden");
        e.currentTarget.classList.toggle("active", !hidden);
      });

      document.getElementById("btn-logout").addEventListener("click", async () => {
        try {
          await this.apiFetch("/homeview/api/auth/logout", { method: "POST", body: {} });
        } catch (_e) {}
        this.csrfToken = "";
        this.sessionToken = "";
        try { sessionStorage.removeItem("hv_sess_tok"); } catch (_e) {}
        this._showLoginScreen();
      });

      // Pin filter chips
      document.querySelectorAll(".hv-filter-chip").forEach(chip => {
        chip.addEventListener("click", () => {
          document.querySelectorAll(".hv-filter-chip").forEach(c => c.classList.remove("active"));
          chip.classList.add("active");
          this.pinFilter = chip.dataset.filter;
        });
      });

      // Zoom & Reset buttons
      document.getElementById("btn-zoom-in").addEventListener("click", () => this.engine?.zoomBy(0.84));
      document.getElementById("btn-zoom-out").addEventListener("click", () => this.engine?.zoomBy(1.18));
      document.getElementById("btn-cam-reset").addEventListener("click", () => {
        this.engine?.setCamera("dollhouse", true);
        document.querySelectorAll("#camera-bar .hv-cam-btn").forEach(b =>
          b.classList.toggle("active", b.dataset.cam === "dollhouse")
        );
      });

      // Popover close buttons
      document.getElementById("popover-close").addEventListener("click", () => {
        document.getElementById("device-popover").classList.add("hidden");
      });
      document.getElementById("weather-pill").addEventListener("click", () => {
        document.getElementById("weather-popover").classList.toggle("hidden");
      });
      document.getElementById("weather-popover-close").addEventListener("click", () => {
        document.getElementById("weather-popover").classList.add("hidden");
      });
      document.getElementById("climate-quick-pill").addEventListener("click", () => {
        const therm = (this.scene?.smartDevices || []).find(d => d.category === "climate");
        if (therm && this.engine) {
          this.engine.focusPoint(therm.x, therm.y, therm.z);
          this._renderDevicePopover(therm);
        }
      });

      // Sidebar tabs
      document.querySelectorAll(".hv-deck-tab").forEach(tab => {
        tab.addEventListener("click", () => {
          document.querySelectorAll(".hv-deck-tab").forEach(t => t.classList.remove("active"));
          document.querySelectorAll(".hv-deck-pane").forEach(p => p.classList.remove("active"));
          tab.classList.add("active");
          document.getElementById(`pane-${tab.dataset.tab}`)?.classList.add("active");
        });
      });

      // Master & Room Light Controls
      document.querySelectorAll("[data-light-action]").forEach(btn => {
        btn.addEventListener("click", () => {
          this.sendAction({
            domain: "lights",
            action: btn.dataset.lightAction,
            target: btn.dataset.lightTarget || "all",
          });
        });
      });
      document.querySelectorAll("[data-room-toggle]").forEach(btn => {
        btn.addEventListener("click", () => {
          this.sendAction({
            domain: "lights",
            action: "toggle",
            target: btn.dataset.roomToggle,
          });
        });
      });

      // Climate Controls
      document.getElementById("btn-temp-down").addEventListener("click", () => this.adjustTemp(-1));
      document.getElementById("btn-temp-up").addEventListener("click", () => this.adjustTemp(1));
      document.querySelectorAll("[data-climate-mode]").forEach(btn => {
        btn.addEventListener("click", () => {
          const temp = this.state?.climate?.target_temperature || 71;
          this.sendAction({ domain: "climate", action: "set", temperature_f: temp, mode: btn.dataset.climateMode });
        });
      });

      // Air Purifier Controls
      document.getElementById("btn-purifier-toggle").addEventListener("click", () => {
        const on = Boolean(this.state?.air_purifier?.is_on);
        this.sendAction({ domain: "air_purifier", action: on ? "off" : "on" });
      });
      document.getElementById("purifier-slider").addEventListener("change", e => {
        const pct = Number(e.target.value);
        document.getElementById("purifier-speed-val").textContent = `${pct}%`;
        this.sendAction({ domain: "air_purifier", action: "speed", percentage: pct });
      });

      // Dishwasher Controls
      document.querySelectorAll("[data-dw-action]").forEach(btn => {
        btn.addEventListener("click", () => {
          this.sendAction({ domain: "dishwasher", action: btn.dataset.dwAction });
        });
      });

      // TV Controls
      document.querySelectorAll("[data-tv-action]").forEach(btn => {
        btn.addEventListener("click", () => {
          this.sendAction({
            domain: "tv",
            action: btn.dataset.tvAction,
            target: "media_player.bedroom_bedroom_tv",
          });
        });
      });

      // Speaker Announce Form
      document.getElementById("speaker-form").addEventListener("submit", e => {
        e.preventDefault();
        const input = document.getElementById("speaker-input");
        const msg = input.value.trim();
        if (!msg) return;
        input.value = "";
        this.sendAction({ domain: "speaker", action: "announce", message: msg });
      });

      // Grocery Form
      document.getElementById("grocery-form").addEventListener("submit", e => {
        e.preventDefault();
        const input = document.getElementById("grocery-input");
        const item = input.value.trim();
        if (!item) return;
        input.value = "";
        this.sendAction({ domain: "groceries", action: "add", item });
      });

      // Home Bot Chat Form
      document.getElementById("chat-form").addEventListener("submit", async e => {
        e.preventDefault();
        const input = document.getElementById("chat-input");
        const sendBtn = document.getElementById("chat-send-btn");
        const msg = input.value.trim();
        if (!msg) return;
        input.value = "";
        sendBtn.disabled = true;
        try {
          const reqId = (crypto.randomUUID ? crypto.randomUUID() : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, c => {
            const r = (Math.random() * 16) | 0;
            return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
          }));
          const res = await this.apiFetch("/homeview/api/chat", {
            method: "POST",
            body: { message: msg, request_id: reqId },
          });
          this.chatMessages = res.messages || this.chatMessages;
          this._renderChat();
          await this.refreshState();
        } catch (err) {
          this.showToast(err.message);
        } finally {
          sendBtn.disabled = false;
        }
      });

      // Kill Switch & Practice Mode
      document.getElementById("btn-kill-switch").addEventListener("click", () => {
        const killed = Boolean(this.state?.kill_switch);
        if (!killed && !window.confirm("Engage Master Kill Switch? This immediately stops Home Bot and locks all device control.")) {
          return;
        }
        this.sendAction({ domain: "kill_switch", action: killed ? "release" : "engage" });
      });

      document.getElementById("btn-practice-mode").addEventListener("click", () => {
        const cur = Boolean(this.state?.practice_mode);
        this.sendAction({ domain: "practice_mode", action: "set", enabled: !cur });
      });
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => App.init());
  } else {
    App.init();
  }
})();
