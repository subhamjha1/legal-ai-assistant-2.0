import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 text-[10px] font-mono font-medium uppercase tracking-wider",
  {
    variants: {
      variant: {
        default: "bg-panel-raised text-parchment-dim border border-hairline-strong",
        brass: "bg-brass-wash text-brass border border-brass/30",
        ledger: "bg-ledger-wash text-ledger border border-ledger/30",
        brick: "bg-brick-wash text-brick border border-brick/30",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant, className }))} {...props} />;
}

export { Badge, badgeVariants };
