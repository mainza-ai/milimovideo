/** Middle-ellipsize long filenames so the extension and leading id stay visible. */
export function middleEllipsis(value: string, max = 42): string {
    if (value.length <= max) return value;
    const dot = value.lastIndexOf('.');
    const ext = dot > 0 && value.length - dot <= 8 ? value.slice(dot) : '';
    const head = Math.ceil((max - ext.length - 1) / 2);
    const tail = Math.floor((max - ext.length - 1) / 2);
    return `${value.slice(0, head)}…${value.slice(value.length - ext.length - tail)}`;
}
