import { clsx } from 'clsx';

interface ToggleProps {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label?: string;
}

export function Toggle({ checked, onChange, label }: ToggleProps) {
    return (
        <button
            onClick={() => onChange(!checked)}
            className="flex items-center gap-2 group"
        >
            <div className={clsx(
                "w-8 h-4 rounded-full relative transition-colors duration-300",
                checked ? "bg-milimo-500" : "bg-white/10"
            )}>
                <div className={clsx(
                    "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform duration-300 shadow-sm",
                    checked ? "left-[18px]" : "left-0.5"
                )} />
            </div>
            {label && <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400 group-hover:text-white transition-colors">{label}</span>}
        </button>
    );
}
