(function () {
  const cards = Array.from(document.querySelectorAll("[data-note-card]"));
  if (!cards.length) {
    return;
  }

  const list = document.querySelector("[data-note-list]");
  const status = document.querySelector("[data-tag-status]");
  const statusName = document.querySelector("[data-tag-status-name]");
  const statusCount = document.querySelector("[data-tag-status-count]");
  const empty = document.querySelector("[data-tag-empty]");
  const buttons = Array.from(document.querySelectorAll("[data-tag-filter]"));
  const labels = {};
  buttons.forEach((button) => {
    if (button.dataset.tagFilter) {
      labels[button.dataset.tagFilter] = button.dataset.tagLabel;
    }
  });

  let current = "";

  function tagFromHash() {
    const match = location.hash.match(/^#tag=([\w-]+)$/);
    return match && labels[match[1]] ? match[1] : "";
  }

  function apply(tag, updateUrl) {
    current = tag;
    let shown = 0;
    cards.forEach((card) => {
      const visible = !tag || card.dataset.tags.split(" ").includes(tag);
      const wasHidden = card.hidden;
      card.hidden = !visible;
      if (visible) {
        shown += 1;
        if (wasHidden || updateUrl) {
          card.classList.remove("is-entering");
          void card.offsetWidth;
          card.classList.add("is-entering");
        }
      }
    });

    buttons.forEach((button) => {
      if (button.classList.contains("tag-status-clear")) {
        return;
      }
      const on = button.dataset.tagFilter === tag;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-pressed", String(on));
      const row = button.closest(".tag-filter--row");
      if (on && row && row.scrollWidth > row.clientWidth) {
        row.scrollLeft = button.offsetLeft - row.offsetLeft - 18;
      }
    });

    if (status) {
      status.hidden = !tag;
      if (tag) {
        statusName.textContent = labels[tag] || tag;
        statusCount.textContent = shown;
      }
    }
    if (empty) {
      empty.hidden = shown > 0;
    }

    if (updateUrl) {
      const url = tag ? `#tag=${tag}` : location.pathname + location.search;
      history.replaceState(null, "", url);
    }
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tag-filter]");
    if (!button) {
      return;
    }
    event.preventDefault();
    const tag = button.dataset.tagFilter;
    const fromCard = Boolean(button.closest("[data-note-card]"));
    // Clicking the active tag in the panel toggles it off; a card chip always selects.
    const next = tag && tag === current && !fromCard ? "" : tag;
    apply(next, true);
    if (fromCard && list) {
      const top = list.getBoundingClientRect().top + window.scrollY - 140;
      if (window.scrollY > top) {
        window.scrollTo({ top, behavior: "smooth" });
      }
    }
  });

  window.addEventListener("hashchange", () => apply(tagFromHash(), false));
  apply(tagFromHash(), false);
})();
