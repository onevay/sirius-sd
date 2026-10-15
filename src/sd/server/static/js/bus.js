// Шина событий интерфейса: экраны подписываются на 'live:alert', 'live:review', 'auth:lost' и т. п.; подписка возвращает функцию отмены.
const handlers = new Map();

export const bus = {
  on(name, fn) {
    if (!handlers.has(name)) handlers.set(name, new Set());
    handlers.get(name).add(fn);
    return () => handlers.get(name)?.delete(fn);
  },
  emit(name, payload) {
    for (const fn of [...(handlers.get(name) || [])]) {
      try { fn(payload); } catch (e) { console.error(`обработчик «${name}» упал`, e); }
    }
  },
};
