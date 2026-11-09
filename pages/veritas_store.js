/**
 * VERITAS Client-Side High-Capacity Storage & Cache Engine
 * Uses IndexedDB to store high-resolution image pairs and verification metrics without hitting localStorage 5MB quota limits.
 * Gracefully falls back to sessionStorage if IndexedDB is unavailable.
 */

(function(global) {
  'use strict';

  const DB_NAME = 'veritas_db';
  const DB_VERSION = 1;
  const STORE_IMAGES = 'image_cache';
  const STORE_METRICS = 'metrics_cache';

  let dbPromise = null;

  function openDB() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve) => {
      if (!('indexedDB' in global)) {
        console.warn('IndexedDB not supported; falling back to sessionStorage.');
        return resolve(null);
      }

      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(STORE_IMAGES)) {
          db.createObjectStore(STORE_IMAGES, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORE_METRICS)) {
          db.createObjectStore(STORE_METRICS, { keyPath: 'id' });
        }
      };

      request.onsuccess = (event) => {
        resolve(event.target.result);
      };

      request.onerror = (event) => {
        console.warn('IndexedDB open error:', event.target.error);
        resolve(null); // Fallback
      };
    });
    return dbPromise;
  }

  const VeritasCache = {
    /**
     * Save the image pair into persistent client cache
     * @param {string} refImage - Base64 or DataURL of reference image
     * @param {string} srcImage - Base64 or DataURL of source/comparative image
     * @param {string} refName - Filename of reference image
     * @param {string} srcName - Filename of source image
     */
    async savePair(refImage, srcImage, refName = 'reference_image', srcName = 'source_image') {
      const payload = {
        id: 'active_pair',
        ref_image: refImage,
        src_image: srcImage,
        ref_name: refName,
        src_name: srcName,
        timestamp: Date.now()
      };

      const db = await openDB();
      if (db) {
        return new Promise((resolve) => {
          try {
            const tx = db.transaction(STORE_IMAGES, 'readwrite');
            const store = tx.objectStore(STORE_IMAGES);
            store.put(payload);
            tx.oncomplete = () => {
              try {
                sessionStorage.setItem('veritas_pair_saved', 'true');
                sessionStorage.setItem('veritas_pair_timestamp', String(payload.timestamp));
              } catch (e) {}
              resolve(true);
            };
            tx.onerror = () => {
              this._fallbackSavePair(payload);
              resolve(true);
            };
          } catch (err) {
            this._fallbackSavePair(payload);
            resolve(true);
          }
        });
      } else {
        this._fallbackSavePair(payload);
        return Promise.resolve(true);
      }
    },

    _fallbackSavePair(payload) {
      try {
        sessionStorage.setItem('veritas_active_pair', JSON.stringify(payload));
      } catch (err) {
        console.warn('sessionStorage pair save error:', err);
      }
    },

    /**
     * Retrieve the active cached image pair
     * @returns {Promise<{ref_image: string, src_image: string, ref_name: string, src_name: string, timestamp: number}|null>}
     */
    async getPair() {
      const db = await openDB();
      if (db) {
        return new Promise((resolve) => {
          try {
            const tx = db.transaction(STORE_IMAGES, 'readonly');
            const store = tx.objectStore(STORE_IMAGES);
            const req = store.get('active_pair');
            req.onsuccess = () => {
              if (req.result) {
                resolve(req.result);
              } else {
                resolve(this._fallbackGetPair());
              }
            };
            req.onerror = () => resolve(this._fallbackGetPair());
          } catch (e) {
            resolve(this._fallbackGetPair());
          }
        });
      } else {
        return Promise.resolve(this._fallbackGetPair());
      }
    },

    _fallbackGetPair() {
      try {
        const raw = sessionStorage.getItem('veritas_active_pair');
        return raw ? JSON.parse(raw) : null;
      } catch (e) {
        return null;
      }
    },

    /**
     * Save verification analysis metrics and reports
     */
    async saveMetrics(metrics) {
      const payload = {
        id: 'latest_metrics',
        data: metrics,
        timestamp: Date.now()
      };

      const db = await openDB();
      if (db) {
        return new Promise((resolve) => {
          try {
            const tx = db.transaction(STORE_METRICS, 'readwrite');
            const store = tx.objectStore(STORE_METRICS);
            store.put(payload);
            tx.oncomplete = () => {
              try {
                sessionStorage.setItem('veritas_metrics_saved', 'true');
              } catch (e) {}
              resolve(true);
            };
            tx.onerror = () => {
              this._fallbackSaveMetrics(payload);
              resolve(true);
            };
          } catch (e) {
            this._fallbackSaveMetrics(payload);
            resolve(true);
          }
        });
      } else {
        this._fallbackSaveMetrics(payload);
        return Promise.resolve(true);
      }
    },

    _fallbackSaveMetrics(payload) {
      try {
        sessionStorage.setItem('veritas_latest_metrics', JSON.stringify(payload.data));
      } catch (e) {
        console.warn('sessionStorage metrics save error:', e);
      }
    },

    /**
     * Retrieve latest verification metrics
     */
    async getMetrics() {
      const db = await openDB();
      if (db) {
        return new Promise((resolve) => {
          try {
            const tx = db.transaction(STORE_METRICS, 'readonly');
            const store = tx.objectStore(STORE_METRICS);
            const req = store.get('latest_metrics');
            req.onsuccess = () => {
              if (req.result) {
                resolve(req.result.data);
              } else {
                resolve(this._fallbackGetMetrics());
              }
            };
            req.onerror = () => resolve(this._fallbackGetMetrics());
          } catch (e) {
            resolve(this._fallbackGetMetrics());
          }
        });
      } else {
        return Promise.resolve(this._fallbackGetMetrics());
      }
    },

    _fallbackGetMetrics() {
      try {
        const raw = sessionStorage.getItem('veritas_latest_metrics');
        return raw ? JSON.parse(raw) : null;
      } catch (e) {
        return null;
      }
    },

    /**
     * Clear all cached images and analysis data
     */
    async clear() {
      try {
        sessionStorage.removeItem('veritas_active_pair');
        sessionStorage.removeItem('veritas_latest_metrics');
        sessionStorage.removeItem('veritas_pair_saved');
      } catch (e) {}

      const db = await openDB();
      if (db) {
        try {
          const tx = db.transaction([STORE_IMAGES, STORE_METRICS], 'readwrite');
          tx.objectStore(STORE_IMAGES).clear();
          tx.objectStore(STORE_METRICS).clear();
        } catch (e) {}
      }
    }
  };

  global.VeritasCache = VeritasCache;
})(typeof window !== 'undefined' ? window : this);
