/**
 * Hansard NLP Explorer - Shared Utilities
 *
 * Common JavaScript functions used across all apps.
 *
 * Usage:
 *   <script src="../shared/hansard-utils.js"></script>
 */

const HansardUtils = {
  // ============================================
  // Constants
  // ============================================

  API_BASE: 'http://localhost:8766',

  COLORS: {
    male: '#3B82C4',
    female: '#EC4899',
    accent: '#8B5CF6',
    success: '#10B981',
    warning: '#F59E0B',
    error: '#EF4444',
  },

  // ============================================
  // Formatting
  // ============================================

  /**
   * Format a number with thousands separators
   */
  formatNumber(num) {
    if (num === null || num === undefined) return '--';
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
  },

  /**
   * Format a number as a full value with commas
   */
  formatNumberFull(num) {
    if (num === null || num === undefined) return '--';
    return num.toLocaleString();
  },

  /**
   * Format a percentage
   */
  formatPercent(value, decimals = 1) {
    if (value === null || value === undefined) return '--%';
    return value.toFixed(decimals) + '%';
  },

  /**
   * Format bytes to human readable
   */
  formatBytes(bytes) {
    if (!bytes) return '0 B';
    if (bytes >= 1024 * 1024 * 1024) return (bytes / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return bytes + ' B';
  },

  /**
   * Format a date string
   */
  formatDate(dateStr) {
    if (!dateStr) return '--';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    });
  },

  /**
   * Truncate text to a maximum length
   */
  truncate(text, maxLength = 100) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength).trim() + '...';
  },

  // ============================================
  // DOM Utilities
  // ============================================

  /**
   * Safely escape HTML to prevent XSS
   */
  escapeHtml(text) {
    if (text === null || text === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
  },

  /**
   * Create an element with attributes and children
   */
  createElement(tag, attrs = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([key, value]) => {
      if (key === 'className') {
        el.className = value;
      } else if (key === 'dataset') {
        Object.entries(value).forEach(([k, v]) => el.dataset[k] = v);
      } else if (key.startsWith('on') && typeof value === 'function') {
        el.addEventListener(key.slice(2).toLowerCase(), value);
      } else {
        el.setAttribute(key, value);
      }
    });
    children.forEach(child => {
      if (typeof child === 'string') {
        el.appendChild(document.createTextNode(child));
      } else if (child) {
        el.appendChild(child);
      }
    });
    return el;
  },

  /**
   * Debounce a function
   */
  debounce(func, wait = 300) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  },

  /**
   * Throttle a function
   */
  throttle(func, limit = 100) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  },

  // ============================================
  // API Helpers
  // ============================================

  /**
   * Make an API request
   */
  async apiRequest(endpoint, params = {}) {
    const url = new URL(`${this.API_BASE}${endpoint}`);
    Object.entries(params).forEach(([key, value]) => {
      if (value !== null && value !== undefined) {
        url.searchParams.set(key, value);
      }
    });

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    return response.json();
  },

  /**
   * Check if API is available
   */
  async checkApiConnection() {
    try {
      const response = await fetch(`${this.API_BASE}/api/stats`);
      return response.ok;
    } catch (e) {
      return false;
    }
  },

  /**
   * Search debates
   */
  async searchDebates(query, options = {}) {
    return this.apiRequest('/api/search', {
      q: query,
      year_from: options.yearFrom,
      year_to: options.yearTo,
      has_female: options.hasFemale,
      gender: options.gender,
      chamber: options.chamber,
      limit: options.limit || 100,
      search_content: options.searchContent ? 'true' : 'false'
    });
  },

  /**
   * Trace a debate through pipeline
   */
  async traceDebate(debateId, year) {
    return this.apiRequest('/api/trace', { id: debateId, year });
  },

  /**
   * Get a specific speech
   */
  async getSpeech(speechId) {
    return this.apiRequest('/api/speech', { id: speechId });
  },

  // ============================================
  // Theme Handling
  // ============================================

  /**
   * Get current theme
   */
  getTheme() {
    return document.body.getAttribute('data-theme') || 'dark';
  },

  /**
   * Set theme
   */
  setTheme(theme) {
    document.body.setAttribute('data-theme', theme === 'light' ? 'light' : '');
    localStorage.setItem('hansard-theme', theme);
  },

  /**
   * Toggle theme between light and dark
   */
  toggleTheme() {
    const current = this.getTheme();
    this.setTheme(current === 'light' ? 'dark' : 'light');
    return this.getTheme();
  },

  /**
   * Load saved theme from localStorage
   */
  loadSavedTheme() {
    const saved = localStorage.getItem('hansard-theme');
    if (saved) {
      this.setTheme(saved);
    }
  },

  // ============================================
  // Speaker Utilities
  // ============================================

  /**
   * Get initials from a name
   */
  getInitials(name) {
    if (!name) return '?';
    return name
      .split(' ')
      .map(w => w[0])
      .filter(c => c && c.match(/[A-Za-z]/))
      .join('')
      .slice(0, 2)
      .toUpperCase() || '?';
  },

  /**
   * Get gender display text
   */
  getGenderLabel(gender) {
    if (gender === 'M') return 'Male';
    if (gender === 'F') return 'Female';
    return 'Unknown';
  },

  /**
   * Get CSS class for gender
   */
  getGenderClass(gender) {
    if (gender === 'M') return 'male';
    if (gender === 'F') return 'female';
    return 'unknown';
  },

  // ============================================
  // Data Utilities
  // ============================================

  /**
   * Group array by key
   */
  groupBy(array, key) {
    return array.reduce((groups, item) => {
      const group = item[key];
      groups[group] = groups[group] || [];
      groups[group].push(item);
      return groups;
    }, {});
  },

  /**
   * Sort array by key
   */
  sortBy(array, key, ascending = true) {
    return [...array].sort((a, b) => {
      const aVal = a[key];
      const bVal = b[key];
      if (aVal < bVal) return ascending ? -1 : 1;
      if (aVal > bVal) return ascending ? 1 : -1;
      return 0;
    });
  },

  /**
   * Count unique values in array
   */
  countUnique(array, key) {
    const seen = new Set();
    array.forEach(item => {
      const val = item[key];
      if (val !== null && val !== undefined) {
        seen.add(val);
      }
    });
    return seen.size;
  },

  // ============================================
  // URL Utilities
  // ============================================

  /**
   * Get query parameters from URL
   */
  getUrlParams() {
    return Object.fromEntries(new URLSearchParams(window.location.search));
  },

  /**
   * Set query parameters in URL
   */
  setUrlParams(params, replace = false) {
    const url = new URL(window.location);
    Object.entries(params).forEach(([key, value]) => {
      if (value === null || value === undefined || value === '') {
        url.searchParams.delete(key);
      } else {
        url.searchParams.set(key, value);
      }
    });
    if (replace) {
      window.history.replaceState({}, '', url);
    } else {
      window.history.pushState({}, '', url);
    }
  },

  // ============================================
  // Export Utilities
  // ============================================

  /**
   * Download data as JSON file
   */
  downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    this.downloadBlob(blob, filename);
  },

  /**
   * Download data as CSV file
   */
  downloadCsv(data, filename) {
    if (!data.length) return;

    const headers = Object.keys(data[0]);
    const rows = [
      headers.join(','),
      ...data.map(row =>
        headers.map(h => {
          const val = row[h];
          if (val === null || val === undefined) return '';
          const str = String(val);
          // Escape quotes and wrap in quotes if contains comma
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return '"' + str.replace(/"/g, '""') + '"';
          }
          return str;
        }).join(',')
      )
    ];

    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    this.downloadBlob(blob, filename);
  },

  /**
   * Download a blob
   */
  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  // ============================================
  // Keyboard Shortcuts
  // ============================================

  /**
   * Register keyboard shortcuts
   */
  registerShortcuts(shortcuts) {
    document.addEventListener('keydown', (e) => {
      // Skip if in input/textarea
      if (e.target.matches('input, textarea, select')) return;

      const key = e.key.toLowerCase();
      const combo = [
        e.ctrlKey || e.metaKey ? 'ctrl' : '',
        e.shiftKey ? 'shift' : '',
        e.altKey ? 'alt' : '',
        key
      ].filter(Boolean).join('+');

      if (shortcuts[combo]) {
        e.preventDefault();
        shortcuts[combo](e);
      }
    });
  }
};

// Export for use as module if supported
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HansardUtils;
}
