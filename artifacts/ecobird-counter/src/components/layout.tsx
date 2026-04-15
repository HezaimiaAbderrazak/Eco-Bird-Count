import { Link, useLocation } from "wouter";
import { Bird, History, BarChart2, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Analysis", icon: Bird },
  { href: "/history", label: "History", icon: History },
  { href: "/stats", label: "Statistics", icon: BarChart2 },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  const NavLinks = () => (
    <>
      {navItems.map((item) => {
        const isActive = location === item.href;
        return (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
              isActive
                ? "bg-primary text-primary-foreground"
                : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
            )}
          >
            <item.icon className="h-5 w-5" />
            {item.label}
          </Link>
        );
      })}
    </>
  );

  return (
    <div className="flex min-h-screen w-full bg-background">
      {/* Desktop Sidebar */}
      <aside className="hidden w-64 flex-col border-r bg-sidebar md:flex">
        <div className="flex h-16 items-center border-b px-6">
          <div className="flex items-center gap-2 font-serif text-lg font-bold text-primary">
            <Bird className="h-6 w-6" />
            <span>EcoBird Counter</span>
          </div>
        </div>
        <nav className="flex-1 space-y-1 p-4">
          <NavLinks />
        </nav>
        <div className="p-4 text-xs text-muted-foreground">
          A field naturalist's digital notebook.
        </div>
      </aside>

      <div className="flex flex-1 flex-col">
        {/* Mobile Header */}
        <header className="flex h-16 items-center border-b bg-background px-4 md:hidden">
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="mr-2 md:hidden">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle navigation menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0">
              <div className="flex h-16 items-center border-b px-6">
                <div className="flex items-center gap-2 font-serif text-lg font-bold text-primary">
                  <Bird className="h-6 w-6" />
                  <span>EcoBird Counter</span>
                </div>
              </div>
              <nav className="space-y-1 p-4">
                <NavLinks />
              </nav>
            </SheetContent>
          </Sheet>
          <div className="flex items-center gap-2 font-serif text-lg font-bold text-primary md:hidden">
            <span>EcoBird</span>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto p-4 md:p-8">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}
